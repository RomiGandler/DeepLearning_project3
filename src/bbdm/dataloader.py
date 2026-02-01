import torch
import torchvision.transforms as transforms
from PIL import Image

from src.bbdm.Register import Registers
from src.data.base_dataloader import BaseChessDataset


class BBDMChessDataset(BaseChessDataset):
    """
    BBDM-compatible dataset inheriting from BaseChessDataset.
    
    Adds BBDM-specific:
        - Horizontal flip augmentation (doubles dataset length)
        - Tensor transforms with torchvision
        - Normalization to [-1, 1] via (x - 0.5) * 2
        
    Return format (matching original BBDM):
        - Without masks: ((x, x_name), (x_cond, x_cond_name))
        - With masks: ((x, x_name), (x_cond, x_cond_name), (mask, mask_name))
        
    Where:
        - x = B (target image)
        - x_cond = A (condition image)
        - Each name is the stem of that specific file's path
    """
    
    def __init__(
        self,
        dataset_config=None,
        stage: str = 'train',
        # Direct config (if no dataset_config):
        dataset_path: str = None,
        image_size: int = 256,
        flip: bool = True,
        to_normal: bool = True,
        use_masks: bool = False,
    ):
        # Extract from BBDM config if provided
        if dataset_config is not None:
            image_size = dataset_config.image_size
            flip = dataset_config.flip if stage == 'train' else False
            to_normal = dataset_config.to_normal
            use_masks = getattr(dataset_config, 'use_masks', False)
            dataset_path = getattr(dataset_config, 'dataset_path', None)
        
        self.to_normal = to_normal
        self.flip = flip if stage == 'train' else False
        self._image_size_tuple = (image_size, image_size)
        
        super().__init__(
            dataset_path=dataset_path,
            stage=stage,
            image_size=image_size,
            use_masks=use_masks,
        )
    
    def __len__(self):
        if self.flip:
            return self._base_length * 2
        return self._base_length
    
    def _transform_image(self, image: Image.Image, flip: bool = False) -> torch.Tensor:
        """Transform PIL image to BBDM-style normalized tensor."""
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        transform = transforms.Compose([
            transforms.Resize(self._image_size_tuple),
            transforms.ToTensor()
        ])
        tensor = transform(image)
        
        if self.to_normal:
            tensor = (tensor - 0.5) * 2.0
            tensor.clamp_(-1., 1.)
        
        return tensor
    
    def _transform_mask(self, mask: Image.Image, flip: bool = False) -> torch.Tensor:
        """Transform mask to tensor (no normalization, kept in [0, 1])."""
        if flip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        transform = transforms.Compose([
            transforms.Resize(self._image_size_tuple),
            transforms.ToTensor()
        ])
        return transform(mask)
    
    def __getitem__(self, index: int):
        # Handle flip augmentation
        should_flip = False
        real_index = index
        if self.flip and index >= self._base_length:
            real_index = index - self._base_length
            should_flip = True
        
        raw = self.get_raw_item(real_index)
        
        image_B = self._transform_image(raw['B'], flip=should_flip)
        image_A = self._transform_image(raw['A'], flip=should_flip)
        filename_B = raw['filename_B']
        filename_A = raw['filename_A']
        
        if self.use_masks and 'mask' in raw:
            mask = self._transform_mask(raw['mask'], flip=should_flip)
            filename_mask = raw['filename_mask']
            # Return format: ((x, x_name), (x_cond, x_cond_name), (mask, mask_name))
            # x = B (target), x_cond = A (condition)
            return (image_B, filename_B), (image_A, filename_A), (mask, filename_mask)
        else:
            # Return format: ((x, x_name), (x_cond, x_cond_name))
            return (image_B, filename_B), (image_A, filename_A)


# Register datasets with BBDM system
# Use 'custom_aligned' and 'masked_aligned' to match existing configs
@Registers.datasets.register_with_name('custom_aligned')
class ChessAlignedDataset(BBDMChessDataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__(dataset_config, stage=stage, use_masks=False)


@Registers.datasets.register_with_name('masked_aligned')
class ChessMaskedDataset(BBDMChessDataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__(dataset_config, stage=stage, use_masks=True)


@Registers.datasets.register_with_name('mask_guided_aligned')
class ChessMaskGuidedDataset(BaseChessDataset):
    """
    Dataset with white/black piece masks for mask-guided conditioning.
    
    Returns: ((x, name), (x_cond, name), (masks_2ch, name))
    Where masks_2ch is [2, H, W] tensor with white mask in channel 0, black in channel 1.
    """
    def __init__(self, dataset_config, stage='train'):
        # Extract params from config
        if dataset_config is not None:
            image_size = dataset_config.image_size
            flip = dataset_config.flip if stage == 'train' else False
            to_normal = dataset_config.to_normal
            dataset_path = getattr(dataset_config, 'dataset_path', None)
            use_mask_guidance = getattr(dataset_config, 'use_mask_guidance', True)
        else:
            image_size = 256
            flip = stage == 'train'
            to_normal = True
            dataset_path = None
            use_mask_guidance = True
        
        self.to_normal = to_normal
        self.flip = flip
        self._image_size_tuple = (image_size, image_size)
        
        super().__init__(
            dataset_path=dataset_path,
            stage=stage,
            image_size=image_size,
            use_masks=False,
            use_mask_guidance=use_mask_guidance,
        )
    
    def __len__(self):
        if self.flip:
            return self._base_length * 2
        return self._base_length
    
    def _transform_image(self, image: Image.Image, flip: bool = False) -> torch.Tensor:
        """Transform PIL image to BBDM-style normalized tensor."""
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        transform = transforms.Compose([
            transforms.Resize(self._image_size_tuple),
            transforms.ToTensor()
        ])
        tensor = transform(image)
        
        if self.to_normal:
            tensor = (tensor - 0.5) * 2.0
            tensor.clamp_(-1., 1.)
        
        return tensor
    
    def _transform_mask(self, mask: Image.Image, flip: bool = False) -> torch.Tensor:
        """Transform mask to tensor (no normalization, kept in [0, 1])."""
        if flip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        transform = transforms.Compose([
            transforms.Resize(self._image_size_tuple),
            transforms.ToTensor()
        ])
        return transform(mask)
    
    def __getitem__(self, index: int):
        # Handle flip augmentation
        should_flip = False
        real_index = index
        if self.flip and index >= self._base_length:
            real_index = index - self._base_length
            should_flip = True
        
        raw = self.get_raw_item(real_index)
        
        image_B = self._transform_image(raw['B'], flip=should_flip)
        image_A = self._transform_image(raw['A'], flip=should_flip)
        filename_B = raw['filename_B']
        filename_A = raw['filename_A']
        
        # Stack white and black masks into 2-channel tensor
        mask_white = self._transform_mask(raw['mask_white'], flip=should_flip)  # [1, H, W]
        mask_black = self._transform_mask(raw['mask_black'], flip=should_flip)  # [1, H, W]
        masks_2ch = torch.cat([mask_white, mask_black], dim=0)  # [2, H, W]
        
        return (image_B, filename_B), (image_A, filename_A), (masks_2ch, filename_A)