"""
SAM-based Grid Extraction with Per-Detection Centroid Assignment.

Unified pipeline that:
1. Runs SAM on an image to detect individual chess pieces
2. For each detection, computes its centroid
3. Classifies each piece as white/black based on pixel intensity
4. Assigns each piece to a grid cell based on its centroid location
"""

import numpy as np
import cv2
import torch
from typing import Tuple, List
from dataclasses import dataclass

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    SAM3SemanticPredictor = None

BOARD_SIZE = 8


@dataclass
class PieceDetection:
    """Represents a single detected chess piece."""
    mask: np.ndarray
    centroid: Tuple[float, float]
    is_white: bool
    area: int
    grid_row: int = -1
    grid_col: int = -1
    filtered: bool = False


class SAMGridExtractor:
    """Extracts chess piece positions as an 8x8 grid using SAM with per-detection centroid logic."""
    
    def __init__(
        self, 
        model_path: str = "/home/avinoamd/roni/BBDM/SAM/sam3.pt",
        device: str = 'auto',
        conf: float = 0.25,
        brightness_threshold: float = 110.0,
        min_area_fraction: float = 0.5,
    ):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"SAMGridExtractor using device: {device}")
        
        self.device = device
        self.brightness_threshold = brightness_threshold
        self.min_area_fraction = min_area_fraction
        self.predictor = None
        
        assert SAM3SemanticPredictor is not None, "ultralytics SAM not available"
        
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=model_path,
            half=(device != 'cpu'),
            save=False,
            device=device
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
    
    def _compute_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
        moments = cv2.moments(mask_uint8)
        
        if moments['m00'] == 0:
            coords = np.argwhere(mask)
            if len(coords) == 0:
                return (0.0, 0.0)
            cy, cx = coords.mean(axis=0)
            return (cx, cy)
        
        return (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
    
    def _classify_piece_color(self, mask: np.ndarray, gray_image: np.ndarray) -> bool:
        masked_pixels = gray_image[mask]
        if len(masked_pixels) == 0:
            return True
        return np.median(masked_pixels) > self.brightness_threshold
    
    def extract_grid(
        self,
        image: np.ndarray,
        board_size: int = BOARD_SIZE,
    ) -> Tuple[np.ndarray, List[PieceDetection]]:
        """
        Extract chess piece grid from an image.
        
        Args:
            image: BGR image as numpy array
            board_size: Size of the chess board grid
            
        Returns:
            Tuple of (grid, detections):
            - grid: (2, 8, 8) array. Channel 0 = White, Channel 1 = Black
            - detections: List of PieceDetection objects
        """
        assert image is not None, "Image is None"
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Run SAM
        self.predictor.set_image(image)
        results = self.predictor(text=["chess piece"])
        
        if not results or results[0].masks is None:
            return np.zeros((2, board_size, board_size), dtype=np.float32), []
        
        masks = results[0].masks.data.cpu().numpy().astype(bool)
        
        # First pass: collect all detections
        raw_detections = []
        for mask in masks:
            area = int(np.sum(mask))
            if area == 0:
                continue
            raw_detections.append(PieceDetection(
                mask=mask,
                centroid=self._compute_centroid(mask),
                is_white=self._classify_piece_color(mask, gray),
                area=area,
            ))
        
        if not raw_detections:
            return np.zeros((2, board_size, board_size), dtype=np.float32), []
        
        # Filter by area
        mean_area = np.mean([d.area for d in raw_detections])
        min_area = self.min_area_fraction * mean_area
        
        for det in raw_detections:
            det.filtered = det.area < min_area
        
        # Build grid from non-filtered detections
        cell_h, cell_w = h / board_size, w / board_size
        grid = np.zeros((2, board_size, board_size), dtype=np.float32)
        
        for det in raw_detections:
            if det.filtered:
                continue
            
            cx, cy = det.centroid
            row = max(0, min(int(cy / cell_h), board_size - 1))
            col = max(0, min(int(cx / cell_w), board_size - 1))
            det.grid_row, det.grid_col = row, col
            
            channel = 0 if det.is_white else 1
            grid[channel, row, col] = 1.0
        
        return grid, raw_detections
