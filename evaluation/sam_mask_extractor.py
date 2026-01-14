"""
Module: SAM-based mask extraction (Evaluation Mode).

Extracts chess piece masks using SAM for validation.
"""

import os
import numpy as np
import cv2
import torch
from typing import Tuple, Optional

# SAM imports
try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    SAM3SemanticPredictor = None
    print("Warning: ultralytics SAM not available. Install with: pip install ultralytics")

class SAMMaskExtractor:
    """
    Extracts chess piece masks using SAM (Segment Anything Model).
    """
    
    DEFAULT_PROMPTS = ["white chess piece", "black chess piece"]
    
    def __init__(
        self, 
        model_path: str = "/home/avinoamd/roni/BBDM/SAM/sam3.pt",
        device: str = 'auto',
        conf: float = 0.25
    ):
        # Auto-detect device if not specified
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"SAM using device: {device}")
        
        self.device = device
        self.model_path = model_path
        self.predictor = None
        
        if SAM3SemanticPredictor is None:
            print("SAM predictor not available")
            return
        
        # Half precision only works on CUDA
        use_half = (device != 'cpu')
            
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=model_path,
            half=use_half,
            save=False,
            device=device
        )
        
        try:
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            print(f"Error initializing SAM predictor: {e}")
            self.predictor = None
    
    def is_available(self) -> bool:
        return self.predictor is not None
    
    def extract_masks(
        self, 
        image_path: str,
        prompts: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract white and black piece masks from an image.
        Returns numpy boolean arrays.
        """
        if not self.is_available():
            raise RuntimeError("SAM predictor not initialized")
        
        if prompts is None:
            prompts = self.DEFAULT_PROMPTS
        
        try:
            self.predictor.set_image(image_path)
            results = self.predictor(text=prompts)
        except Exception as e:
            raise RuntimeError(f"Error processing image {image_path}: {e}")
        
        if not results:
            img = cv2.imread(image_path)
            if img is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            h, w = img.shape[:2]
            return np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)
        
        result = results[0]
        
        if result.masks is not None:
            h, w = result.masks.data.shape[1:]
        else:
            h, w = result.orig_shape
        
        mask_white = np.zeros((h, w), dtype=bool)
        mask_black = np.zeros((h, w), dtype=bool)
        
        if result.masks is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy().astype(bool)
            
            for i, cls_id in enumerate(classes):
                if cls_id == 0:
                    mask_white = np.logical_or(mask_white, masks[i])
                elif cls_id == 1:
                    mask_black = np.logical_or(mask_black, masks[i])
        
        return mask_white, mask_black
    
    def save_debug_masks(
        self,
        mask_white: np.ndarray,
        mask_black: np.ndarray,
        output_dir: str,
        file_id: str
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(output_dir, f"{file_id}_white_mask.png"),
            mask_white.astype(np.uint8) * 255
        )
        cv2.imwrite(
            os.path.join(output_dir, f"{file_id}_black_mask.png"),
            mask_black.astype(np.uint8) * 255
        )
