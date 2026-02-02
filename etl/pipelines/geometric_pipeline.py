#!/usr/bin/env python3
"""
Pipeline 3: Geometric Augmentation

This pipeline applies D4 group geometric transformations to all images in a directory.

Transformations (excluding identity):
- rot90: 90 degrees clockwise rotation
- rot180: 180 degrees rotation
- rot270: 270 degrees clockwise rotation
- flip_h: Horizontal flip
- flip_v: Vertical flip
- flip_diag: Main diagonal flip (transpose)
- flip_antidiag: Anti-diagonal flip

For each input image, creates 7 transformed copies with suffixes:
  original.png -> original_rot90.png, original_rot180.png, etc.

Usage:
    python -m etl.pipelines.geometric_pipeline \
        --input_dir etl/test_data/A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from etl.augmentations.geometric import (
    get_transforms,
    apply_transform,
    add_transform_suffix,
    has_transform_suffix,
    TRANSFORM_NAMES,
)


def get_image_files(input_dir: Path, skip_transformed: bool = True) -> list[Path]:
    """
    Get list of image files from a directory.

    Args:
        input_dir: Directory to scan
        skip_transformed: If True, skip files that already have transform suffixes

    Returns:
        List of image file paths
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if skip_transformed:
        image_files = [f for f in image_files if not has_transform_suffix(f.name)]

    return sorted(image_files)


def process_single_image(
    image_path: Path,
    output_dir: Path,
    include_identity: bool = False,
) -> list[Path]:
    """
    Apply all geometric transforms to a single image.

    Args:
        image_path: Path to input image
        output_dir: Output directory for augmented images
        include_identity: If True, also save identity (unchanged) copy

    Returns:
        List of paths to saved augmented images
    """
    # Load image
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except Exception as e:
        print(f"  ERROR: Failed to load image: {e}")
        return []

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get transforms
    transforms = get_transforms(include_identity=include_identity)

    # Apply and save each transform
    saved_paths = []
    for name, transform_func in transforms.items():
        try:
            aug_array = transform_func(img_array)
            aug_img = Image.fromarray(aug_array)

            out_name = add_transform_suffix(image_path.name, name)
            out_path = output_dir / out_name

            aug_img.save(out_path)
            saved_paths.append(out_path)
        except Exception as e:
            print(f"    ERROR: Transform '{name}' failed: {e}")

    return saved_paths


def run_pipeline(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    include_identity: bool = False,
    skip_transformed: bool = True,
) -> bool:
    """
    Run the complete geometric augmentation pipeline.

    Args:
        input_dir: Directory containing images to process
        output_dir: Output directory (defaults to input_dir for in-place)
        include_identity: If True, also create identity (unchanged) copies
        skip_transformed: If True, skip images that already have transform suffixes

    Returns:
        True if successful
    """
    print("=" * 60)
    print("Pipeline 3: Geometric Augmentation")
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

    # Show transforms being applied
    transforms = get_transforms(include_identity=include_identity)
    print(f"\nTransforms to apply ({len(transforms)}):")
    for name in transforms.keys():
        print(f"  - {name}")

    # Get image files
    print(f"\n[Step 1] Scanning for images in: {input_dir}")
    image_files = get_image_files(input_dir, skip_transformed=skip_transformed)
    print(f"  Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print("  No images to process (all may already have transform suffixes)")
        return True

    expected_outputs = len(image_files) * len(transforms)
    print(f"  Expected output: {expected_outputs} augmented images")

    # Process images
    print(f"\n[Step 2] Applying geometric transforms...")
    total_saved = 0
    total_failed = 0

    for i, image_path in enumerate(image_files, 1):
        print(f"\n  [{i}/{len(image_files)}] {image_path.name}")

        saved_paths = process_single_image(
            image_path,
            output_dir,
            include_identity=include_identity,
        )

        total_saved += len(saved_paths)
        failed = len(transforms) - len(saved_paths)
        total_failed += failed

        if failed > 0:
            print(f"    WARNING: {failed} transforms failed")
        else:
            print(f"    Saved {len(saved_paths)} augmented images")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print(f"  Input images: {len(image_files)}")
    print(f"  Transforms per image: {len(transforms)}")
    print(f"  Total saved: {total_saved} augmented images")
    if total_failed > 0:
        print(f"  Total failed: {total_failed}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)

    return total_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Geometric Augmentation Pipeline (D4 group transforms)"
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
        "--include_identity",
        action="store_true",
        help="Also create identity (unchanged) copies"
    )
    parser.add_argument(
        "--include_transformed",
        action="store_true",
        help="Process images that already have transform suffixes"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    success = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        include_identity=args.include_identity,
        skip_transformed=not args.include_transformed,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
