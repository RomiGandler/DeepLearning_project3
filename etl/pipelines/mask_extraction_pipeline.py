#!/usr/bin/env python3
"""
Pipeline 4: Mask Extraction

This pipeline extracts black and white piece masks from chess board images.

For each image in input_dir:
1. Detects chess pieces using SAM
2. Clusters pieces into black/white groups by grayscale intensity
3. Saves combined black piece mask to black_output_dir
4. Saves combined white piece mask to white_output_dir

Output masks are binary images (0 or 255) with the same filename as input.

Usage:
    python -m etl.pipelines.mask_extraction_pipeline \
        --input_dir chess_data/train/B/ \
        --black_output_dir chess_data/train/mask_black/ \
        --white_output_dir chess_data/train/mask_white/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from etl.augmentations.mask_extraction import SAMMaskExtractor


def get_image_files(input_dir: Path) -> list[Path]:
    """
    Get list of image files from a directory.

    Args:
        input_dir: Directory to scan

    Returns:
        List of image file paths, sorted by name
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    return sorted(image_files)


def run_pipeline(
    input_dir: Path,
    black_output_dir: Path,
    white_output_dir: Path,
    device: str = "auto",
    conf: float = 0.4,
) -> bool:
    """
    Run the mask extraction pipeline.

    Args:
        input_dir: Directory containing images to process
        black_output_dir: Output directory for black piece masks
        white_output_dir: Output directory for white piece masks
        device: Device for SAM ("auto", "cuda", "cpu")
        conf: SAM confidence threshold

    Returns:
        True if all images processed successfully
    """
    print("=" * 60)
    print("Pipeline 4: Mask Extraction")
    print("=" * 60)

    # Validate input directory
    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        return False

    print(f"Input: {input_dir}")
    print(f"Black masks output: {black_output_dir}")
    print(f"White masks output: {white_output_dir}")

    # Get image files
    print(f"\n[Step 1] Scanning for images...")
    image_files = get_image_files(input_dir)
    print(f"  Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print("  No images found")
        return True

    # Create output directories
    black_output_dir.mkdir(parents=True, exist_ok=True)
    white_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SAM (expensive)
    print(f"\n[Step 2] Initializing SAM model...")
    print(f"  Device: {device}")
    print(f"  Confidence: {conf}")

    try:
        extractor = SAMMaskExtractor(device=device, conf=conf)
    except Exception as e:
        print(f"  ERROR: Failed to initialize SAM: {e}")
        return False

    # Process images
    print(f"\n[Step 3] Extracting masks...")
    success_count = 0
    fail_count = 0

    for i, image_path in enumerate(image_files, 1):
        print(f"\n  [{i}/{len(image_files)}] {image_path.name}")

        # Output paths use the same filename
        black_output_path = black_output_dir / image_path.name
        white_output_path = white_output_dir / image_path.name

        success = extractor.extract_and_save(
            image_path=image_path,
            black_output_path=black_output_path,
            white_output_path=white_output_path,
        )

        if success:
            print(f"    Saved: {black_output_path.name} (black), {white_output_path.name} (white)")
            success_count += 1
        else:
            print(f"    FAILED: No masks extracted")
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print(f"  Processed: {success_count} images")
    if fail_count > 0:
        print(f"  Failed: {fail_count} images")
    print(f"  Black masks: {black_output_dir}")
    print(f"  White masks: {white_output_dir}")
    print("=" * 60)

    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Mask Extraction Pipeline - Extract black/white piece masks from chess images"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--black_output_dir",
        type=str,
        required=True,
        help="Output directory for black piece masks"
    )
    parser.add_argument(
        "--white_output_dir",
        type=str,
        required=True,
        help="Output directory for white piece masks"
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

    args = parser.parse_args()

    success = run_pipeline(
        input_dir=Path(args.input_dir),
        black_output_dir=Path(args.black_output_dir),
        white_output_dir=Path(args.white_output_dir),
        device=args.device,
        conf=args.conf,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
