import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset

from .hf_downloader import HFResourceManager


def get_image_paths_from_dir(fdir: str) -> List[str]:
    """Recursively get all image paths from a directory."""
    flist = sorted(os.listdir(fdir))
    image_paths = []
    for fname in flist:
        fpath = os.path.join(fdir, fname)
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                image_paths.append(fpath)
    return image_paths


class BaseChessDataset(Dataset, ABC):
    """
    Abstract base dataset for chess board images.
    
    All chess dataloaders (BBDM, VQGAN) must inherit from this class.
    
    If dataset_path is provided, uses it directly (with validation).
    If dataset_path is None, automatically downloads from HuggingFace.
    
    Directory structure expected:
        dataset_path/
        ├── train/
        │   ├── A/              # Condition images
        │   ├── B/              # Target images
        │   ├── masks/          # Optional single-channel masks (legacy)
        │   ├── A_mask_white/   # Optional white piece masks
        │   └── A_mask_black/   # Optional black piece masks
        ├── val/
        └── test/
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        stage: str = 'train',
        image_size: int = 256,
        use_masks: bool = True,
        use_mask_guidance: bool = False,
    ):
        """
        Args:
            dataset_path: Local path to dataset. If None, downloads from HuggingFace.
            stage: One of 'train', 'val', 'test'
            image_size: Target image size (square)
            use_masks: Whether to load single-channel mask images (legacy)
            use_mask_guidance: Whether to load white/black piece masks for guidance
        """
        super().__init__()
        
        # Resolve dataset path
        self.dataset_path = self._resolve_dataset_path(dataset_path)
        
        self.stage = stage
        self.image_size = image_size
        self.use_masks = use_masks
        self.use_mask_guidance = use_mask_guidance
        
        # Load and validate paths
        self._load_paths()
    
    def _resolve_dataset_path(self, dataset_path: Optional[str]) -> Path:
        """
        Resolve dataset path.
        
        If path provided: validate and use it.
        If None: download from HuggingFace.
        """
        if dataset_path is not None:
            path = Path(dataset_path)
            assert path.exists(), f"Dataset path does not exist: {dataset_path}"
            return path
        
        # No path provided - download from HuggingFace
        hf_manager = HFResourceManager()
        return hf_manager.get_dataset()
    
    def _load_paths(self):
        """Load and validate image paths from directories."""
        stage_path = self.dataset_path / self.stage
        
        assert stage_path.exists(), f"Stage directory does not exist: {stage_path}"
        assert (stage_path / 'A').exists(), f"Missing A/ directory in {stage_path}"
        assert (stage_path / 'B').exists(), f"Missing B/ directory in {stage_path}"
        
        self.paths_A = get_image_paths_from_dir(str(stage_path / 'A'))
        self.paths_B = get_image_paths_from_dir(str(stage_path / 'B'))
        
        # Optional single-channel masks (legacy)
        mask_dir = stage_path / 'masks'
        if self.use_masks:
            if not mask_dir.exists():
                raise FileNotFoundError(
                    f"use_masks=True but masks/ directory not found at {mask_dir}. "
                    f"Either set use_masks=False or create the masks/ directory."
                )
            self.paths_masks = get_image_paths_from_dir(str(mask_dir))
        else:
            self.paths_masks = None
        
        # Optional white/black piece masks for guidance
        self.paths_mask_white = None
        self.paths_mask_black = None
        if self.use_mask_guidance:
            white_dir = stage_path / 'A_mask_white'
            black_dir = stage_path / 'A_mask_black'
            if not white_dir.exists() or not black_dir.exists():
                raise FileNotFoundError(
                    f"use_mask_guidance=True but A_mask_white/ or A_mask_black/ not found in {stage_path}. "
                    f"Either set use_mask_guidance=False or create both directories."
                )
            self.paths_mask_white = get_image_paths_from_dir(str(white_dir))
            self.paths_mask_black = get_image_paths_from_dir(str(black_dir))
        
        # Validate alignment
        assert len(self.paths_A) == len(self.paths_B), \
            f"Mismatch: {len(self.paths_A)} A images vs {len(self.paths_B)} B images"
        assert len(self.paths_A) > 0, f"No images found in {stage_path}"
        
        if self.paths_masks:
            assert len(self.paths_A) == len(self.paths_masks), \
                f"Mismatch: {len(self.paths_A)} images vs {len(self.paths_masks)} masks"
        
        if self.paths_mask_white:
            assert len(self.paths_A) == len(self.paths_mask_white), \
                f"Mismatch: {len(self.paths_A)} images vs {len(self.paths_mask_white)} white masks"
            assert len(self.paths_A) == len(self.paths_mask_black), \
                f"Mismatch: {len(self.paths_A)} images vs {len(self.paths_mask_black)} black masks"
        
        self._base_length = len(self.paths_A)
    
    def __len__(self):
        return self._base_length
    
    def _load_image(self, path: str) -> Image.Image:
        """Load RGB image."""
        image = Image.open(path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def _load_mask(self, path: str) -> Image.Image:
        """Load grayscale mask."""
        mask = Image.open(path)
        if mask.mode != 'L':
            mask = mask.convert('L')
        return mask
    
    def get_raw_item(self, index: int) -> Dict[str, Any]:
        """
        Get raw PIL images before framework-specific transforms.
        
        Returns:
            dict with keys: 
                - 'A': PIL image from A/ directory (condition)
                - 'B': PIL image from B/ directory (target)
                - 'filename_A': stem of A image filename
                - 'filename_B': stem of B image filename
                - 'path_A': full path to A image
                - 'path_B': full path to B image
                - 'mask' (if use_masks): PIL image from masks/ directory
                - 'filename_mask' (if use_masks): stem of mask filename
                - 'mask_white' (if use_mask_guidance): PIL image of white piece mask
                - 'mask_black' (if use_mask_guidance): PIL image of black piece mask
        """
        result = {
            'A': self._load_image(self.paths_A[index]),
            'B': self._load_image(self.paths_B[index]),
            'filename_A': Path(self.paths_A[index]).stem,
            'filename_B': Path(self.paths_B[index]).stem,
            'path_A': self.paths_A[index],
            'path_B': self.paths_B[index],
        }
        
        if self.use_masks and self.paths_masks:
            result['mask'] = self._load_mask(self.paths_masks[index])
            result['filename_mask'] = Path(self.paths_masks[index]).stem
        
        if self.use_mask_guidance and self.paths_mask_white:
            result['mask_white'] = self._load_mask(self.paths_mask_white[index])
            result['mask_black'] = self._load_mask(self.paths_mask_black[index])
        
        return result
    
    @abstractmethod
    def __getitem__(self, index: int):
        """Subclasses must implement to return framework-specific format."""
        raise NotImplementedError