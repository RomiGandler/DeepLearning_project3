#!/usr/bin/env python3
"""
Color augmentation script for chess board images.

Uses SAM to detect chess pieces, clusters them into black/white groups,
then swaps the colors using Reinhard color transfer to preserve texture and edges.

The transfer works by:
1. Computing mean and std RGB for each cluster (black/white pieces)
2. For each pixel: offset = (pixel - source_mean) / source_std
3. New pixel = target_mean + offset * target_std

This preserves the relative color variations (texture, edges, shading) within
each piece while shifting the overall color distribution to the opposite cluster.

Output:
1. identity - original image
2. colors_swapped - black pieces become white-colored and vice versa (texture preserved)
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

from src.evaluation.model_loader import get_sam_checkpoint


def cluster_piece_colors_2means(
    masks: np.ndarray,
    gray_image: np.ndarray,
    brightness_threshold: float = 110.0,
) -> tuple[np.ndarray, float, float]:
    """
    Cluster per-mask mean grayscale intensity into 2 clusters (2-means).
    
    Args:
        masks: Boolean mask array of shape (N, H, W)
        gray_image: Grayscale image (H, W)
        brightness_threshold: Fallback threshold if clustering fails
        
    Returns:
        Tuple of (labels, center_black, center_white):
        - labels: Array of 0 (black) or 1 (white) per mask
        - center_black: Grayscale intensity of black cluster center
        - center_white: Grayscale intensity of white cluster center
    """
    if masks is None or len(masks) == 0:
        return np.array([]), 0.0, 255.0

    # Feature: mean grayscale intensity per mask
    intensities = []
    for m in masks:
        pixels = gray_image[m]
        intensities.append(float(np.mean(pixels)) if len(pixels) else 0.0)

    # If we don't have enough detections to cluster, fall back to thresholding
    if len(intensities) < 2 or np.allclose(intensities, intensities[0]):
        labels = np.array([1 if i > brightness_threshold else 0 for i in intensities])
        return labels, brightness_threshold / 2, brightness_threshold * 1.5

    x = np.asarray(intensities, dtype=np.float32)

    # Initialize centers as min/max (robust for 1D)
    c0 = float(np.min(x))
    c1 = float(np.max(x))

    # Run a small fixed-iteration 2-means in 1D
    for _ in range(25):
        d0 = np.abs(x - c0)
        d1 = np.abs(x - c1)
        labels = (d1 < d0).astype(np.int32)  # 0 -> c0, 1 -> c1

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
    # Return labels as 0=black, 1=white
    if c0 > c1:
        # c0 is white, c1 is black -> flip labels
        labels = 1 - labels
        return labels, c1, c0
    else:
        # c0 is black, c1 is white
        return labels, c0, c1


def compute_cluster_rgb_stats(
    masks: np.ndarray,
    labels: np.ndarray,
    rgb_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and std RGB color for each cluster (for Reinhard color transfer).
    
    Args:
        masks: Boolean mask array of shape (N, H, W)
        labels: Array of 0 (black) or 1 (white) per mask
        rgb_image: RGB image (H, W, 3)
        
    Returns:
        Tuple of (black_mean, black_std, white_mean, white_std) each as RGB array (3,)
    """
    black_pixels = []
    white_pixels = []
    
    for mask, label in zip(masks, labels):
        pixels = rgb_image[mask]  # Shape: (num_pixels, 3)
        if len(pixels) == 0:
            continue
        if label == 0:  # black cluster
            black_pixels.append(pixels)
        else:  # white cluster
            white_pixels.append(pixels)
    
    # Compute mean and std RGB for each cluster
    if black_pixels:
        black_all = np.concatenate(black_pixels, axis=0).astype(np.float32)
        black_mean = np.mean(black_all, axis=0)
        black_std = np.std(black_all, axis=0)
        black_std = np.maximum(black_std, 1.0)  # avoid division by zero
    else:
        black_mean = np.array([50, 50, 50], dtype=np.float32)
        black_std = np.array([20, 20, 20], dtype=np.float32)
    
    if white_pixels:
        white_all = np.concatenate(white_pixels, axis=0).astype(np.float32)
        white_mean = np.mean(white_all, axis=0)
        white_std = np.std(white_all, axis=0)
        white_std = np.maximum(white_std, 1.0)  # avoid division by zero
    else:
        white_mean = np.array([200, 200, 200], dtype=np.float32)
        white_std = np.array([20, 20, 20], dtype=np.float32)
    
    return black_mean, black_std, white_mean, white_std


def swap_piece_colors(
    image: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    black_mean: np.ndarray,
    black_std: np.ndarray,
    white_mean: np.ndarray,
    white_std: np.ndarray,
) -> np.ndarray:
    """
    Create a new image with piece colors swapped using Reinhard color transfer.
    
    Preserves texture and edges by transferring each pixel's deviation from its
    cluster mean to the target cluster, scaled by the std ratio.
    
    For a white piece pixel:
        offset = (pixel - white_mean) / white_std
        new_pixel = black_mean + offset * black_std
    
    Args:
        image: RGB image (H, W, 3)
        masks: Boolean mask array of shape (N, H, W)
        labels: Array of 0 (black) or 1 (white) per mask
        black_mean: Mean RGB of black cluster (3,)
        black_std: Std RGB of black cluster (3,)
        white_mean: Mean RGB of white cluster (3,)
        white_std: Std RGB of white cluster (3,)
        
    Returns:
        New image with texture-preserving swapped colors
    """
    result = image.astype(np.float32).copy()
    
    for mask, label in zip(masks, labels):
        pixels = result[mask]  # Shape: (num_pixels, 3)
        
        if label == 0:
            # Black piece -> transfer to white color distribution
            # Normalize by black stats, then apply white stats
            offset = (pixels - black_mean) / black_std
            new_pixels = white_mean + offset * white_std
        else:
            # White piece -> transfer to black color distribution
            offset = (pixels - white_mean) / white_std
            new_pixels = black_mean + offset * black_std
        
        result[mask] = new_pixels
    
    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


class SAMColorSwapper:
    """Uses SAM to detect chess pieces and swap black/white colors."""
    
    def __init__(self, device: str = "auto", conf: float = 0.4):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"SAMColorSwapper using device: {device}")
        
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
    
    def process(self, image_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """
        Process an image and return identity + color-swapped versions.
        
        Args:
            image_bgr: BGR image (OpenCV format)
            
        Returns:
            Dict with 'identity' and 'colors_swapped' RGB images
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Run SAM to get masks
        self.predictor.set_image(image_bgr)
        results = self.predictor(text=["chess piece"])
        
        if not results or results[0].masks is None:
            print("Warning: No pieces detected, returning identity only")
            return {"identity": rgb, "colors_swapped": rgb}
        
        masks = results[0].masks.data.cpu().numpy().astype(bool)
        print(f"Detected {len(masks)} pieces")
        
        # Cluster into black/white
        labels, _, _ = cluster_piece_colors_2means(masks, gray)
        n_black = np.sum(labels == 0)
        n_white = np.sum(labels == 1)
        print(f"Clustered: {n_black} black, {n_white} white pieces")
        
        # Compute RGB stats for Reinhard color transfer
        black_mean, black_std, white_mean, white_std = compute_cluster_rgb_stats(
            masks, labels, rgb
        )
        print(f"Black cluster - mean: {black_mean.astype(int)}, std: {black_std.astype(int)}")
        print(f"White cluster - mean: {white_mean.astype(int)}, std: {white_std.astype(int)}")
        
        # Create swapped image with texture-preserving color transfer
        swapped = swap_piece_colors(
            rgb, masks, labels, black_mean, black_std, white_mean, white_std
        )
        
        return {"identity": rgb, "colors_swapped": swapped}


def main():
    # === MODIFY THESE ===
    IMAGE_PATH = None  # Path to the input chess board image
    OUTPUT_DIR = None  # Output directory (default: directory named after image stem + "_colors")
    FORMAT = None      # Output format (default: same as input)
    # ====================

    # Load image
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Determine output directory
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = image_path.parent / f"{image_path.stem}_colors"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output format
    output_format = FORMAT or image_path.suffix.lstrip(".")
    if output_format.lower() == "jpg":
        output_format = "jpeg"
    
    # Process
    print(f"Input image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    swapper = SAMColorSwapper()
    results = swapper.process(image_bgr)
    
    print(f"Generating {len(results)} augmentations...")
    for name, rgb_array in results.items():
        # Convert back to BGR for OpenCV saving
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        output_path = output_dir / f"{name}.{output_format}"
        cv2.imwrite(str(output_path), bgr_array)
        print(f"  Saved: {output_path.name}")
    
    print("Done!")


if __name__ == "__main__":
    main()
