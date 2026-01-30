#!/usr/bin/env python3
"""
Geometric augmentation script for chess board images.

Creates all 8 distinct geometric transformations of a square (dihedral group D4):
1. identity - no transformation
2. rot90 - 90° clockwise rotation
3. rot180 - 180° rotation
4. rot270 - 270° clockwise rotation
5. flip_h - horizontal flip (mirror over vertical axis)
6. flip_v - vertical flip (mirror over horizontal axis)
7. flip_diag - main diagonal flip (transpose)
8. flip_antidiag - anti-diagonal flip
"""

from pathlib import Path

import numpy as np
from PIL import Image


def get_augmentations(img: np.ndarray) -> dict[str, np.ndarray]:
    """
    Generate all 8 geometric augmentations of a square image.
    
    Args:
        img: Input image as numpy array (H, W, C) or (H, W)
    
    Returns:
        Dictionary mapping augmentation names to transformed images
    """
    augmentations = {
        # Identity
        "identity": img,
        # Rotations (k=1 is 90° counter-clockwise, so we use -k for clockwise)
        "rot90": np.rot90(img, k=-1),  # 90° clockwise
        "rot180": np.rot90(img, k=2),   # 180°
        "rot270": np.rot90(img, k=1),   # 270° clockwise (= 90° counter-clockwise)
        # Flips
        "flip_h": np.fliplr(img),       # Horizontal flip
        "flip_v": np.flipud(img),       # Vertical flip
        # Diagonal flips
        # Main diagonal: flip across top-left to bottom-right (transpose)
        "flip_diag": np.swapaxes(img, 0, 1),
        # Anti-diagonal: flip across top-right to bottom-left (transpose + 180°)
        "flip_antidiag": np.rot90(np.swapaxes(img, 0, 1), k=2)
    }
    return augmentations


def main():
    # === MODIFY THESE ===
    IMAGE_PATH = "/Users/avinoamd/Downloads/game12_frame_013912.png"  # Path to the input chess board image
    OUTPUT_DIR = None  # Output directory (default: directory named after the image stem)
    FORMAT = None      # Output format (default: same as input)
    # ====================

    # Load image
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Determine output directory
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = image_path.parent / image_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output format
    output_format = FORMAT or image_path.suffix.lstrip(".")
    if output_format.lower() == "jpg":
        output_format = "jpeg"
    
    # Generate and save augmentations
    augmentations = get_augmentations(img_array)
    
    print(f"Input image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Generating {len(augmentations)} augmentations...")
    
    for name, aug_array in augmentations.items():
        aug_img = Image.fromarray(aug_array)
        output_path = output_dir / f"{name}.{output_format}"
        aug_img.save(output_path)
        print(f"  Saved: {output_path.name}")
    
    print("Done!")


if __name__ == "__main__":
    main()
