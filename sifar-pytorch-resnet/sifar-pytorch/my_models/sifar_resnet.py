import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetForImageClassification
from einops import rearrange

def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

class ResNet50ActionRecognition(nn.Module):
    def __init__(self, num_classes=400, duration=8, super_img_rows=1, img_size=224):
        """
        Args:
            num_classes (int): Number of classes for classification.
            duration (int): Number of frames per video clip.
            super_img_rows (int): Number of rows in the super image. 
                                  If >1, frames will be rearranged into a super image.
                                  For example, if duration=24 and super_img_rows=4, the
                                  resulting super image will have 4 rows and 6 columns.
            img_size (int): Target size (height and width) for each frame.
        """
        super(ResNet50ActionRecognition, self).__init__()
        self.duration = duration
        self.super_img_rows = super_img_rows
        self.img_size = img_size

        # For super image creation, we follow a similar strategy to sifar_swin:
        self.image_mode = True  # We always create a super image if super_img_rows > 1
        # Compute frame padding so that duration is divisible by super_img_rows
        self.frame_padding = self.duration % self.super_img_rows
        if self.frame_padding != 0:
            self.frame_padding = self.super_img_rows - self.frame_padding
            self.duration += self.frame_padding

        # Load the pretrained ResNet-50 from Hugging Face, ignoring mismatched sizes.
        self.resnet = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            ignore_mismatched_sizes=True
        )
        # Replace the classifier head with one that outputs the desired number of classes.
        in_features = self.resnet.classifier[1].in_features
        self.resnet.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Provide a default configuration for normalization.
        self.default_cfg = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }

    def pad_frames(self, x):
        """
        If the number of frames T is less than the adjusted duration,
        pad with zeros along the temporal dimension.
        Then, flatten the temporal dimension into the channel dimension.
        Input shape: (B, T, 3, H, W)
        Output shape: (B, 3 * duration, H, W)
        """
        B, T, C, H, W = x.shape
        if T < self.duration:
            pad_frames = self.duration - T
            pad_tensor = torch.zeros(B, pad_frames, C, H, W, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=1)
        # Flatten temporal dimension into channels: (B, T, 3, H, W) -> (B, 3*T, H, W)
        x = x.view(B, 3 * self.duration, H, W)
        return x

    def create_super_img(self, x):
        """
        Converts a tensor of shape (B, 3*T, H, W) into a super image.
        The T frames are rearranged into a grid with super_img_rows rows.
        The number of columns is (T // super_img_rows).
        
        Returns a tensor of shape: (B, 3, super_img_rows*H, (T//super_img_rows)*W)
        """
        B, C_times_T, H, W = x.shape
        T = C_times_T // 3
        r = self.super_img_rows
        if T % r != 0:
            # Should not occur because we padded in pad_frames.
            raise ValueError("Number of frames is not divisible by super_img_rows.")
        cols = T // r
        # Rearrange: from (B, 3*T, H, W) to (B, 3, r*H, (T//r)*W)
        super_img = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=r, tw=cols, c=3)
        # Debug print removed; now we return the super image.
       
        """
        print("Shape of the super image = ", super_img.shape)   [batch, 3, 672, 672]
        from torchvision.utils import save_image

        # Extract the first image (shape: [3, 672, 672])
        single_image = super_img[0]

        # Save the image to a file without applying any normalization.
        save_image(single_image, "/scratch/workspace/sudipta/saveimage/New_super_image.png", normalize=False)

        exit(0)

        """

        return super_img

    def forward(self, x):
        """
        Preprocesses the input and returns the logits.
        
        Supports:
          - 5D inputs (video clips) of shape (B, T, 3, H, W)
          - 4D inputs (single images) of shape (B, 3, H, W) or (B, H, W, 3)
        
        For 5D inputs:
          - If super_img_rows > 1, create a super image.
          - Otherwise, use the middle frame.
        For 4D inputs:
          - If channels are last, permute to (B, 3, H, W).
          - If the number of channels is a multiple of 3, treat it as a video clip.
        """
        if x.dim() == 5:
            # Video clip input: shape (B, T, 3, H, W)
            if self.super_img_rows > 1:
                # Pad frames if needed.
                if x.size(1) < self.duration or self.frame_padding > 0:
                    x = self.pad_frames(x)
                else:
                    B, T, C, H, W = x.shape
                    x = x.view(B, 3 * T, H, W)
                # Create the super image.
                x = self.create_super_img(x)
            else:
                # Use the middle frame.
                mid = x.size(1) // 2
                x = x[:, mid, ...]
        elif x.dim() == 4:
            # Single image input.
            if x.shape[1] == 3:
                pass  # Already (B, 3, H, W)
            elif x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            elif x.shape[1] % 3 == 0:
                # Treat stacked channels as a video clip.
                T = x.shape[1] // 3
                x = x.view(x.shape[0], T, 3, x.shape[2], x.shape[3])
                if self.super_img_rows > 1:
                    if x.size(1) < self.duration or self.frame_padding > 0:
                        x = self.pad_frames(x)
                    else:
                        B, T, C, H, W = x.shape
                        x = x.view(B, 3 * T, H, W)
                    x = self.create_super_img(x)
                else:
                    mid = x.size(1) // 2
                    x = x[:, mid, ...]
            else:
                raise ValueError("4D input does not have 3 channels and is not a multiple of 3.")
        else:
            raise ValueError(f"Expected input tensor with 4 or 5 dimensions, got {x.dim()}.")
        
        # Now x should be of shape (B, 3, H_new, W_new). Pass it through the ResNet.
        return self.resnet(x).logits

def build_resnet50(num_classes=400, duration=8, super_img_rows=1, img_size=224):
    return ResNet50ActionRecognition(num_classes=num_classes, duration=duration, super_img_rows=super_img_rows, img_size=img_size)
