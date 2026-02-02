"""
Crop chessboard area from rendered Blender images.

Crops rendered images to the chessboard area only (removes frame/background).
"""

from __future__ import annotations

import cv2
import os
import argparse
import glob
from pathlib import Path
from typing import Optional

# ==========================================
# Cropping calibration parameters
# ==========================================
# Adjust these values until the red rectangle (in preview mode)
# perfectly aligns with the chessboard area only (without frame or background).

CROP_Y_START = 75   # Top boundary
CROP_Y_END   = 725  # Bottom boundary

CROP_X_START = 75   # Left boundary
CROP_X_END   = 725  # Right boundary
# ==========================================


def process_single_image(
    image_path: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    preview_mode: bool = False,
) -> Optional[str]:
    """
    Process a single image:
    - In preview mode: draw a red rectangle for calibration.
    - In action mode: crop the image according to the defined boundaries.

    Args:
        image_path: Path to the image to process
        output_dir: Optional output directory (defaults to overwriting original)
        output_name: Optional output filename (defaults to original filename)
        preview_mode: If True, just draw crop rectangle for calibration

    Returns:
        Path to saved file, or None if preview mode or error
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return None

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image {image_path}")
        return None

    # === Preview mode: draw red rectangle only ===
    if preview_mode:
        cv2.rectangle(
            img,
            (CROP_X_START, CROP_Y_START),
            (CROP_X_END, CROP_Y_END),
            (0, 0, 255),  # Red color in BGR
            2
        )

        preview_name = "preview_calibration.png"
        cv2.imwrite(preview_name, img)
        print(f"Preview mode: saved calibration image to '{preview_name}'")
        print("Verify the red box alignment and adjust crop values if needed.")
        return None

    # === Action mode: perform actual cropping ===
    cropped_img = img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

    # Determine save path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = output_name if output_name else os.path.basename(image_path)
        save_path = os.path.join(output_dir, filename)
    else:
        # Overwrite original image
        save_path = image_path

    cv2.imwrite(save_path, cropped_img)
    print(
        f"Cropped and saved: {save_path} "
        f"(New size: {cropped_img.shape[1]}x{cropped_img.shape[0]})"
    )
    return save_path


def crop_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    preview_mode: bool = False,
) -> list[str]:
    """
    Crop all PNG images in a directory.

    Args:
        input_dir: Directory containing images
        output_dir: Optional output directory
        preview_mode: If True, just show crop preview for first image

    Returns:
        List of saved file paths
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Directory {input_dir} not found")
        return []

    images = glob.glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(images)} images. Processing...")

    saved_paths = []
    for img_path in images:
        result = process_single_image(img_path, output_dir, preview_mode=preview_mode)
        if result:
            saved_paths.append(result)
        if preview_mode:
            break  # Only preview first image

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Crop chessboard area from images.")

    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--output_dir", type=str, help="Optional output directory")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview crop area by drawing a red rectangle (no cropping)"
    )

    args = parser.parse_args()

    if args.image:
        process_single_image(args.image, args.output_dir, preview_mode=args.preview)

    elif args.dir:
        crop_directory(args.dir, args.output_dir, preview_mode=args.preview)
    else:
        print("Usage: python -m etl.blender.crop_board --image <path> [--preview]")


if __name__ == "__main__":
    main()
