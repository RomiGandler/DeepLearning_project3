"""
Module: SAM-based mask extraction (Evaluation Mode).

Extracts chess piece masks using SAM for validation.
"""

import numpy as np
import cv2
import torch
from typing import Tuple

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
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract white and black piece masks from an image using a two-step process:
        1. Detect ALL chess pieces using a generic prompt.
        2. Classify each piece as white/black based on pixel intensity.
        
        Args:
            image: BGR image as numpy array (from cv2.imread)
        """
        assert self.is_available(), "SAM predictor not initialized"
        assert image is not None, "Image is None"
        
        detection_prompts = ["chess piece"]
        
        self.predictor.set_image(image)
        results = self.predictor(text=detection_prompts)
        
        h, w = image.shape[:2]
        
        mask_white = np.zeros((h, w), dtype=bool)
        mask_black = np.zeros((h, w), dtype=bool)

        if not results:
            return mask_white, mask_black
        
        result = results[0]
        
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy().astype(bool)
            
            # Convert image to grayscale for intensity check
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            for i in range(len(masks)):
                single_mask = masks[i]
                
                # Get pixels belonging to this mask
                masked_pixels = gray_img[single_mask]
                
                if len(masked_pixels) == 0:
                    continue
                
                # Calculate median brightness to be robust against highlights/shadows
                avg_brightness = np.median(masked_pixels)
                
                # Threshold can be tuned. 110 is a good starting point.
                if avg_brightness > 110: 
                    mask_white = np.logical_or(mask_white, single_mask)
                else:
                    mask_black = np.logical_or(mask_black, single_mask)
        
        return mask_white, mask_black
