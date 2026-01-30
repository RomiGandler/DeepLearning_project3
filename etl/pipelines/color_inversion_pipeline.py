#!/usr/bin/env python3
"""
Pipeline 2: Color Inversion (SAM-based)

This pipeline:
1. Loads SAMColorSwapper model (expensive, done once)
2. For each image in the input directory:
   - Detects chess pieces using SAM
   - Clusters pieces into black/white groups
   - Applies Reinhard color transfer to swap colors
   - Saves color-swapped image with "_inv" suffix
3. Skips images that already have "_inv" suffix

Usage:
    python -m etl.pipelines.color_inversion_pipeline \
        --input_dir chess_data/train/B/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2

from etl.augmentations.color_swap import SAMColorSwapper
from etl.augmentations.fen_utils import add_inv_suffix, has_inv_suffix


def get_image_files(input_dir: Path, skip_inverted: bool = True) -> list[Path]:
    """
    Get list of image files from a directory.

    Args:
        input_dir: Directory to scan
        skip_inverted: If True, skip files with "_inv" suffix

    Returns:
        List of image file paths
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if skip_inverted:
        image_files = [f for f in image_files if not has_inv_suffix(f.name)]

    return sorted(image_files)


def process_single_image(
    image_path: Path,
    swapper: SAMColorSwapper,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Process a single image and save color-swapped version.

    Args:
        image_path: Path to input image
        swapper: SAMColorSwapper instance
        output_dir: Output directory (defaults to same as input)

    Returns:
        Path to saved swapped image, or None if failed
    """
    # Load image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  ERROR: Failed to load image: {image_path}")
        return None

    # Process with SAM color swapper
    try:
        swapped_rgb = swapper.swap_colors(image_bgr)
    except Exception as e:
        print(f"  ERROR: Color swap failed: {e}")
        return None

    # Convert back to BGR for saving
    swapped_bgr = cv2.cvtColor(swapped_rgb, cv2.COLOR_RGB2BGR)

    # Determine output path
    out_dir = output_dir if output_dir else image_path.parent
    out_name = add_inv_suffix(image_path.name)
    out_path = out_dir / out_name

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    cv2.imwrite(str(out_path), swapped_bgr)
    return out_path


def run_pipeline(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    device: str = "auto",
    conf: float = 0.4,
    skip_inverted: bool = True,
) -> bool:
    """
    Run the complete color inversion pipeline.

    Args:
        input_dir: Directory containing images to process
        output_dir: Output directory (defaults to input_dir)
        device: Device for SAM ("auto", "cuda", "cpu")
        conf: SAM confidence threshold
        skip_inverted: If True, skip images with "_inv" suffix

    Returns:
        True if successful
    """
    print("=" * 60)
    print("Pipeline 2: Color Inversion (SAM-based)")
    print("=" * 60)

    # Validate input directory
    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        return False

    # Set output directory
    if output_dir is None:
        output_dir = input_dir
        print(f"Output: In-place (same directory)")
    else:
        print(f"Output: {output_dir}")

    # Get image files
    print(f"\n[Step 1] Scanning for images in: {input_dir}")
    image_files = get_image_files(input_dir, skip_inverted=skip_inverted)
    print(f"  Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print("  No images to process (all may already have _inv suffix)")
        return True

    # Initialize SAM (expensive)
    print(f"\n[Step 2] Initializing SAM model...")
    print(f"  Device: {device}")
    print(f"  Confidence: {conf}")

    try:
        swapper = SAMColorSwapper(device=device, conf=conf)
    except Exception as e:
        print(f"  ERROR: Failed to initialize SAM: {e}")
        return False

    # Process images
    print(f"\n[Step 3] Processing images...")
    success_count = 0
    fail_count = 0

    for i, image_path in enumerate(image_files, 1):
        print(f"\n  [{i}/{len(image_files)}] {image_path.name}")

        result = process_single_image(image_path, swapper, output_dir)
        if result:
            print(f"    Saved: {result.name}")
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print(f"  Processed: {success_count} images")
    if fail_count > 0:
        print(f"  Failed: {fail_count} images")
    print("=" * 60)

    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Color Inversion Pipeline (SAM-based)"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for SAM model (default: auto)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="SAM confidence threshold (default: 0.4)"
    )
    parser.add_argument(
        "--include_inverted",
        action="store_true",
        help="Process images that already have _inv suffix"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    success = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        device=args.device,
        conf=args.conf,
        skip_inverted=not args.include_inverted,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
