"""
SAM-based mask extraction for chess piece images.

Uses SAM (Segment Anything Model) to detect chess pieces, clusters them into
black/white groups based on grayscale intensity, then outputs separate binary
masks for each color group.

This module reuses the clustering logic from color_swap.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

from etl.augmentations.color_swap import cluster_piece_colors_2means
from etl.utils.checkpoint_utils import get_sam_checkpoint


def combine_masks_by_label(
    masks: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine individual piece masks into unified black/white masks.

    Args:
        masks: Boolean mask array of shape (N, H, W)
        labels: Array of 0 (black) or 1 (white) per mask

    Returns:
        Tuple of (black_mask, white_mask), each of shape (H, W) as bool arrays
    """
    if len(masks) == 0:
        raise ValueError("No masks provided")

    h, w = masks.shape[1], masks.shape[2]

    black_mask = np.zeros((h, w), dtype=bool)
    white_mask = np.zeros((h, w), dtype=bool)

    for mask, label in zip(masks, labels):
        if label == 0:
            black_mask |= mask
        else:
            white_mask |= mask

    return black_mask, white_mask


def mask_to_image(mask: np.ndarray) -> np.ndarray:
    """
    Convert a boolean mask to a uint8 image (0 or 255).

    Args:
        mask: Boolean mask of shape (H, W)

    Returns:
        uint8 image where True -> 255, False -> 0
    """
    return (mask.astype(np.uint8) * 255)


class SAMMaskExtractor:
    """
    Uses SAM to detect chess pieces and extract black/white piece masks.

    This class:
    1. Loads SAM model once (expensive initialization)
    2. For each image, detects chess pieces using semantic segmentation
    3. Clusters detected pieces into black/white groups by grayscale intensity
    4. Returns separate binary masks for black and white pieces
    """

    def __init__(self, device: str = "auto", conf: float = 0.4):
        """
        Initialize SAM-based mask extractor.

        Args:
            device: Device to run SAM on ("auto", "cuda", "cpu")
            conf: Confidence threshold for piece detection
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"SAMMaskExtractor using device: {device}")

        self.device = device
        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=get_sam_checkpoint(),
            half=(device != "cpu"),
            save=False,
            device=device,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def extract_masks(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract black and white piece masks from an image.

        Args:
            image_bgr: BGR image (OpenCV format)

        Returns:
            Tuple of (black_mask, white_mask), each of shape (H, W) as bool arrays.
            Returns (None, None) if no pieces are detected.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Run SAM to get masks
        self.predictor.set_image(image_bgr)
        results = self.predictor(text=["chess piece"])

        if not results or results[0].masks is None:
            print("Warning: No pieces detected")
            return None, None

        masks = results[0].masks.data.cpu().numpy().astype(bool)
        print(f"Detected {len(masks)} pieces")

        # Cluster into black/white
        labels, center_black, center_white = cluster_piece_colors_2means(masks, gray)
        
        if len(labels) == 0:
            print("Warning: No pieces to cluster")
            return None, None

        n_black = np.sum(labels == 0)
        n_white = np.sum(labels == 1)
        print(f"Clustered: {n_black} black, {n_white} white pieces")
        print(f"Cluster centers - black: {center_black:.1f}, white: {center_white:.1f}")

        # Combine masks by label
        black_mask, white_mask = combine_masks_by_label(masks, labels)

        return black_mask, white_mask

    def extract_and_save(
        self,
        image_path: Path,
        black_output_path: Path,
        white_output_path: Path,
    ) -> bool:
        """
        Extract masks from an image and save to specified paths.

        Args:
            image_path: Path to input image
            black_output_path: Path to save black pieces mask
            white_output_path: Path to save white pieces mask

        Returns:
            True if successful, False if no pieces detected
        """
        # Load image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"ERROR: Failed to load image: {image_path}")
            return False

        # Extract masks
        black_mask, white_mask = self.extract_masks(image_bgr)

        if black_mask is None or white_mask is None:
            return False

        # Ensure output directories exist
        black_output_path.parent.mkdir(parents=True, exist_ok=True)
        white_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to images and save
        black_img = mask_to_image(black_mask)
        white_img = mask_to_image(white_mask)

        cv2.imwrite(str(black_output_path), black_img)
        cv2.imwrite(str(white_output_path), white_img)

        return True
