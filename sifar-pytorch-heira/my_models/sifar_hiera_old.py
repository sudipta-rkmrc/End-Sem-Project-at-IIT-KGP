# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import math
from functools import partial
from typing import List, Tuple, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import DropPath, Mlp

from .hiera_utils import pretrained_model, conv_nd, do_pool, do_masked_conv, Unroll, Reroll
from .hfhub import has_config, PyTorchModelHubMixin

def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x
 
class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = x.shape



        # Check and handle cases where q_stride or window_size is a tuple
        # print(f"q_stride: {self.q_stride}, window_size: {self.window_size}")

        # Handle q_stride as a tuple if it is one
        if isinstance(self.q_stride, tuple):
            self.q_stride = self.q_stride[0]  # You can also choose another strategy for handling tuples

        # Handle window_size as a tuple if it is one
        if isinstance(self.window_size, tuple):
            window_size_x, window_size_y = self.window_size
            num_windows = (N // (self.q_stride * window_size_x * window_size_y)) if self.use_mask_unit_attn else 1
        else:
            num_windows = (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            #print("Line 115 = ",x.device)
            #print("Line 115 Q = ",q.device)
            #print("Line 115 K = ",k.device)
            #print("Line 115 V = ",v.device)
            #exit(0)
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = (attn @ v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)

        # Interpolate or resize tensors to match their shape
        attn_output = self.attn(x_norm)

        # Use interpolation to resize the attention output to match the shape of x
        # Resize the second dimension of attn_output to match x's second dimension (28224)
        # We assume that attn_output has shape [batch_size, num_tokens, dim_out]
        
        # Interpolate along the second dimension (height or tokens) to match x's shape
        attn_output_resized = F.interpolate(
            attn_output.permute(0, 2, 1),  # Permute to [batch_size, dim_out, num_tokens] for interpolation
            size=x.shape[1],  # Match the second dimension (number of tokens)
            mode='nearest'
        ).permute(0, 2, 1)  # Revert back to [batch_size, num_tokens, dim_out]

        # Now add the tensors after resizing
        x = x + self.drop_path(attn_output_resized)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.softmax(dim=-1),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.projection = nn.Linear(dim, num_classes)
        # act_fun for eval and testing only
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x


class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class Hiera(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224),
        img_size=224,
        duration=8,
        in_chans: int = 3,
        embed_dim: int = 96,
        num_heads: int = 1,
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_pos_embed: bool = False,
        super_img_rows: int = 1,
        **kwargs
    ):
        super().__init__()
        self.default_cfg = {
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5),
        }
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.image_mode = True
        self.duration = duration
        self.img_size = img_size
        self.super_img_rows = super_img_rows
        self.patch_stride = patch_stride
        self.tokens_spatial_shape = [i // s for i, s in zip(input_size, patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)

        # Frame padding for super images
        self.frame_padding = self.duration % super_img_rows if self.image_mode else 0
        if self.frame_padding != 0:
            self.frame_padding = super_img_rows - self.frame_padding
            self.duration += self.frame_padding

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_kernel, patch_stride, patch_padding)

        # Positional embedding
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.duration, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(stages))]

        # Define stage_ends
        self.stage_ends = [sum(stages[:i + 1]) - 1 for i in range(len(stages))]

        # Construct blocks
        self.blocks = nn.ModuleList()
        cur_stage = 0
        for i in range(sum(stages)):
            dim_out = embed_dim
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=q_stride if i in self.stage_ends[:q_pool] else 1,
                window_size=mask_unit_size[0],
                use_mask_unit_attn=use_mask_unit_attn,
            )
            embed_dim = dim_out
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, self.duration, embed_dim))
        nn.init.trunc_normal_(self.frame_pos_embed, std=0.02)

        # Initialize weights
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return ["pos_embed_spatial", "pos_embed_temporal"] if self.sep_pos_embed else ["pos_embed"]

    def create_super_img(self, x):
        """Create a super image by arranging frames into a grid."""
        input_size = x.shape[-2:]
        if input_size != (self.img_size, self.img_size):
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')
        x = rearrange(x, 'b (th tw c) h w -> b c (th h) (tw w)', th=self.super_img_rows, c=3)
        
        """
        print("Shape of the x = ",x.shape)
        import torch
        import torchvision.utils as vutils

        # Assuming `images` is your tensor of shape [32, 3, 672, 672]
        # Select one image (e.g., the first image)
        image = x[1]  # Shape: [3, 672, 672]

        # Save the image
        #vutils.save_image(image, "/scratch/workspace/sudipta/saveimage/saved_image_heira1.png")

        
        exit(0)
        """
        
        return x

    def pad_frames(self, x):
        """Pad frames to match the duration."""
        frame_num = self.duration - self.frame_padding
        x = x.view((-1, 3 * frame_num) + x.size()[2:])
        x_padding = torch.zeros((x.shape[0], 3 * self.frame_padding) + x.size()[2:]).cuda()
        x = torch.cat((x, x_padding), dim=1)
        assert x.shape[1] == 3 * self.duration, f"Frame number {x.shape[1]} != {3 * self.duration}"
        #print("Shape of the x = ",x.shape)
        #exit(0)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Handle frame padding and super image creation
        #print(x.device)

        
        if self.frame_padding > 0:
            x = self.pad_frames(x)
        else:
            x = x.view((-1, 3 * self.duration) + x.size()[2:])

        if self.super_img_rows > 1:
            x = self.create_super_img(x)
            #print(x.device)

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial + self.pos_embed_temporal
        else:
            pos_embed = self.pos_embed

        # Interpolate pos_embed to match the spatial dimensions of x
        if pos_embed.shape[1] != x.shape[1]:
            # Reshape pos_embed to 2D (assuming 2D spatial dimensions)
            pos_embed = pos_embed.reshape(1, self.tokens_spatial_shape[0], self.tokens_spatial_shape[1], -1).permute(0, 3, 1, 2)
            # Interpolate to match the spatial dimensions of x
            pos_embed = F.interpolate(pos_embed, size=(int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))), mode='bilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[1])

        #print(f"pos_embed shape after interpolation: {pos_embed.shape}")
        #print(f"x shape: {x.shape}")
        #exit(0)

        x = x + pos_embed
        #print(x.device)

        # Forward through blocks
        for block in self.blocks:
            x = block(x)
        #print(x.device)
        #exit(0)
        # Normalization and head
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        return x
# Image models

@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_tiny_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_small_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_base_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_base_plus_224(**kwdargs):
    return Hiera(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_large_224(**kwdargs):
    return Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_huge_224(**kwdargs):
    return Hiera(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs)


# Video models

@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_base_16x224(num_classes: int = 400, **kwdargs):
    return Hiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_base_plus_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_large_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_huge_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs
    )