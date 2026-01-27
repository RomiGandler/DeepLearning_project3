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
from src.evaluation.grid_extractors.interface import PieceDetection, BOARD_SIZE, GridExtractor
from src.evaluation.model_loader import get_sam_checkpoint

try:
    from ultralytics.models.sam import SAM3SemanticPredictor
except ImportError:
    SAM3SemanticPredictor = None


class SAMGridExtractor(GridExtractor):
    """Extracts chess piece positions as an 8x8 grid using SAM with per-detection centroid logic."""
    
    def __init__(
        self, 
        device: str = 'auto',
        conf: float = 0.4,
        brightness_threshold: float = 110.0,
        min_area_fraction: float = 0,
    ):
        """
        Initialize SAM-based grid extractor.
        
        The SAM model (sam3.pt) is automatically downloaded from HuggingFace
        if not found in the local checkpoints directory.
        
        Args:
            device: Device to run on ('auto', 'cuda', or 'cpu')
            conf: Confidence threshold for SAM detections
            brightness_threshold: Threshold for white/black piece classification
            min_area_fraction: Minimum area fraction to filter small detections
        """
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"SAMGridExtractor using device: {device}")
        
        self.device = device
        self.brightness_threshold = brightness_threshold
        self.min_area_fraction = min_area_fraction
        self.predictor = None
        
        assert SAM3SemanticPredictor is not None, "ultralytics SAM not available"
        
        # Get SAM model path (downloads from HuggingFace if needed)
        model_path = get_sam_checkpoint()
        
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=str(model_path),
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

    def _cluster_piece_colors(self, masks: np.ndarray, gray_image: np.ndarray) -> List[bool]:
        """
        Cluster per-mask mean grayscale intensity into 2 clusters (2-means) and
        return per-mask `is_white` assignment. The brighter cluster is 'white'.
        """
        if masks is None or len(masks) == 0:
            return []

        # Feature: mean grayscale intensity per mask
        intensities: List[float] = []
        for m in masks:
            pixels = gray_image[m]
            intensities.append(float(np.mean(pixels)) if len(pixels) else 0.0)

        # If we don't have enough detections to cluster, fall back to thresholding
        if len(intensities) < 2 or np.allclose(intensities, intensities[0]):
            return [float(i) > self.brightness_threshold for i in intensities]

        x = np.asarray(intensities, dtype=np.float32)

        # Initialize centers as min/max (robust for 1D)
        c0 = float(np.min(x))
        c1 = float(np.max(x))

        # Run a small fixed-iteration 2-means in 1D
        for _ in range(25):
            # Assign to closest center
            d0 = np.abs(x - c0)
            d1 = np.abs(x - c1)
            labels = (d1 < d0).astype(np.int32)  # 0 -> c0, 1 -> c1

            # Recompute centers; handle empty cluster by keeping previous center
            if np.any(labels == 0):
                new_c0 = float(np.mean(x[labels == 0]))
            else:
                new_c0 = c0
            if np.any(labels == 1):
                new_c1 = float(np.mean(x[labels == 1]))
            else:
                new_c1 = c1

            if abs(new_c0 - c0) < 1e-3 and abs(new_c1 - c1) < 1e-3:
                c0, c1 = new_c0, new_c1
                break
            c0, c1 = new_c0, new_c1

        # Identify which cluster is white (higher center)
        white_label = 0 if c0 > c1 else 1
        return [bool(lbl == white_label) for lbl in labels.tolist()]
    
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
        is_white_list = self._cluster_piece_colors(masks, gray)
        
        # First pass: collect all detections
        raw_detections = []
        for mask, is_white in zip(masks, is_white_list):
            area = int(np.sum(mask))
            if area == 0:
                continue
            raw_detections.append(PieceDetection(
                mask=mask,
                centroid=self._compute_centroid(mask),
                is_white=is_white,
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
