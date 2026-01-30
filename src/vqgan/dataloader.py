import numpy as np
from PIL import Image
from typing import Dict, Any

from src.data.base_dataloader import BaseChessDataset


class VQGANChessDataset(BaseChessDataset):
    """
    VQGAN-compatible dataset inheriting from BaseChessDataset.
    
    Adds VQGAN-specific:
        - Albumentations preprocessing (rescale + crop)
        - Normalization to [-1, 1] via (x / 127.5 - 1.0)
        - Return format: {"image": np.ndarray, "file_path_": str}
    
    VQGAN trains on single images (typically B targets), not pairs.
    """
    
    def __init__(
        self,
        size: int = 256,
        dataset_path: str = None,
        stage: str = 'train',
        image_key: str = 'B',  # 'A', 'B', or 'both'
        random_crop: bool = False,
    ):
        self.image_key = image_key
        self._size = size
        self.random_crop = random_crop
        
        super().__init__(
            dataset_path=dataset_path,
            stage=stage,
            image_size=size,
            use_masks=False,
        )
        
        # Build VQGAN-specific paths list
        self._build_paths()
    
    def _build_paths(self):
        """Build paths list based on image_key setting."""
        if self.image_key == 'both':
            self.paths = self.paths_A + self.paths_B
        elif self.image_key == 'A':
            self.paths = self.paths_A
        else:  # 'B' (default)
            self.paths = self.paths_B
        
        self._vqgan_length = len(self.paths)
    
    def __len__(self):
        return self._vqgan_length
    
    def _preprocess_image(self, image) -> np.ndarray:
        """Preprocess PIL image VQGAN-style."""
        image = image.resize((self._size, self._size), Image.BILINEAR)
        arr = np.array(image).astype(np.uint8)
        # VQGAN normalization: [0, 255] -> [-1, 1]
        return (arr / 127.5 - 1.0).astype(np.float32)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Returns VQGAN-compatible format."""
        path = self.paths[index]
        image = self._load_image(path)
        
        return {
            "image": self._preprocess_image(image),
            "file_path_": path,
        }



class VQGANChessTrain(VQGANChessDataset):
    """Training dataset for VQGAN (matches taming-transformers interface)."""
    
    def __init__(
        self,
        size: int,
        dataset_path: str = None,
        image_key: str = 'B',
    ):
        super().__init__(
            size=size,
            dataset_path=dataset_path,
            stage='train',
            image_key=image_key,
            random_crop=False,
        )


class VQGANChessVal(VQGANChessDataset):
    """Validation dataset for VQGAN."""
    
    def __init__(
        self,
        size: int,
        dataset_path: str = None,
        image_key: str = 'B',
    ):
        super().__init__(
            size=size,
            dataset_path=dataset_path,
            stage='val',
            image_key=image_key,
            random_crop=False,
        )


class VQGANChessTest(VQGANChessDataset):
    """Test dataset for VQGAN."""
    
    def __init__(
        self,
        size: int,
        dataset_path: str = None,
        image_key: str = 'B',
    ):
        super().__init__(
            size=size,
            dataset_path=dataset_path,
            stage='test',
            image_key=image_key,
            random_crop=False,
        )