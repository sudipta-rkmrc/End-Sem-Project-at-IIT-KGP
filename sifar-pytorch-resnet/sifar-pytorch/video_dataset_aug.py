"""



import multiprocessing
from typing import Union, List, Tuple

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from video_transforms import (GroupRandomHorizontalFlip, GroupOverSample,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)

def get_augmentor(is_train: bool, image_size: int, mean: List[float] = None,
                  std: List[float] = None, disable_scaleup: bool = False,
                  threed_data: bool = False, version: str = 'v1', scale_range: [int] = None,
                  modality: str = 'rgb', num_clips: int = 1, num_crops: int = 1, dataset: str = ''):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range

    if modality == 'sound':
        augments = [
            Stack(threed_data=threed_data),
            ToTorchFormatTensor(div=False, num_clips_crops=num_clips * num_crops)
        ]
    else:
        augments = []
        if is_train:
            if version == 'v1':
                augments += [
                    GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
                ]
            elif version == 'v2':
                augments += [
                    GroupRandomScale(scale_range),
                    GroupRandomCrop(image_size),
                ]
            if not (dataset.startswith('ststv') or 'jester' in dataset or 'mini_ststv' in dataset):
                augments += [GroupRandomHorizontalFlip(is_flow=(modality == 'flow'))]
        else:
            scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
            if num_crops == 1:
                augments += [
                    GroupScale(scaled_size),
                    GroupCenterCrop(image_size)
                ]
            else:
                flip = True if num_crops == 10 else False
                augments += [
                    GroupOverSample(image_size, scaled_size, num_crops=num_crops, flip=flip),
                ]
        augments += [
            Stack(threed_data=threed_data),
            ToTorchFormatTensor(num_clips_crops=num_clips * num_crops),
            GroupNormalize(mean=mean, std=std, threed_data=threed_data)
        ]

    augmentor = transforms.Compose(augments)
    return augmentor


def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader


    
    """

import multiprocessing
from typing import List

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from video_transforms import (
    GroupRandomHorizontalFlip, GroupScale, GroupCenterCrop, GroupRandomCrop,
    GroupNormalize, Stack, ToTorchFormatTensor
)

class GroupRandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        """
        Applies the same random resized crop to all frames in a video clip.
        :param size: Target output size (e.g., 224 for ResNet-50).
        :param scale: Range of size of the cropped area before resizing.
        :param ratio: Aspect ratio range of the cropped area.
        """
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img_group):
        """
        Apply the same crop to all frames in the video.
        """
        # Choose random crop parameters based on the first frame
        i, j, h, w = transforms.RandomResizedCrop.get_params(img_group[0], self.scale, self.ratio)
        
        # Apply crop and resize to all frames
        return [transforms.functional.resized_crop(img, i, j, h, w, (self.size, self.size)) for img in img_group]

def get_augmentor(is_train: bool, image_size: int, mean: list = None, std: list = None, disable_scaleup: bool = False,
                  threed_data: bool = False, version: str = 'v1', scale_range: list = None,
                  modality: str = 'rgb', num_clips: int = 1, num_crops: int = 1, dataset: str = ''):
    # your existing code here...

    """
    Returns the data augmentation pipeline based on training or validation mode.
    """
    
    # Set default normalization values (standard for ImageNet-trained ResNet-50)
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std

    augments = []

    if modality == 'sound':
        # Sound processing (if needed)
        augments += [
            Stack(threed_data=False),
            ToTorchFormatTensor(div=False, num_clips_crops=num_clips * num_crops)
        ]
    else:
        # Training Augmentation
        if is_train:
            augments += [
                GroupRandomResizedCrop(image_size, scale=(0.08, 1.0)),  # ResNet-style crop
                GroupRandomHorizontalFlip(),  # Standard horizontal flipping
            ]
        else:
            # Validation Augmentation
            augments += [
                GroupScale(256),  # Resize smaller side to 256
                GroupCenterCrop(image_size),  # Center crop to 224×224
            ]

        # Convert to tensor and normalize
        augments += [
            Stack(threed_data=False),
            ToTorchFormatTensor(num_clips_crops=num_clips * num_crops),
            GroupNormalize(mean=mean, std=std, threed_data=False)
        ]

    augmentor = transforms.Compose(augments)
    return augmentor


def build_dataflow(dataset, is_train, batch_size, workers=8, is_distributed=False):
    """
    Creates a DataLoader for training or validation.
    """
    workers = min(workers, multiprocessing.cpu_count())  # Limit workers to CPU count
    shuffle = is_train  # Shuffle only for training

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None  # Enable shuffle if no distributed sampler

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader
