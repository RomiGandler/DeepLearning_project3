#!/usr/bin/env python3
"""
Pipeline 1: FEN Inversion + Blender Generation

This pipeline:
1. Loads a CSV with FEN and IMG_NAME columns
2. Creates inverted duplicate rows (swap black/white pieces in FEN)
3. Saves extended CSV with "_with_invs" suffix
4. Generates synthetic images via Blender for all rows
5. Crops all generated images to board area

Usage:
    python -m etl.pipelines.fen_inversion_pipeline \
        --csv chess_data/train/gt.csv \
        --output chess_data/train/A/ \
        --blender_path /path/to/blender \
        --blend_file blender/chess-set.blend
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from etl.augmentations.fen_utils import invert_fen_colors, add_inv_suffix
from etl.blender.crop_board import crop_directory


def extend_csv_with_inversions(
    df: pd.DataFrame,
    fen_column: str = "FEN",
    img_name_column: str = "IMG_NAME",
) -> pd.DataFrame:
    """
    Extend a DataFrame by adding inverted FEN duplicates.

    For each row, creates a duplicate with:
    - Inverted FEN (black <-> white pieces)
    - IMG_NAME with "_inv" suffix

    Args:
        df: Input DataFrame
        fen_column: Column name for FEN strings
        img_name_column: Column name for image filenames

    Returns:
        Extended DataFrame with original + inverted rows
    """
    # Create inverted rows
    inverted_rows = []
    for _, row in df.iterrows():
        new_row = row.copy()
        new_row[fen_column] = invert_fen_colors(row[fen_column])
        new_row[img_name_column] = add_inv_suffix(row[img_name_column])
        inverted_rows.append(new_row)

    # Combine original + inverted
    inverted_df = pd.DataFrame(inverted_rows)
    extended_df = pd.concat([df, inverted_df], ignore_index=True)

    return extended_df


def save_extended_csv(original_path: Path, df: pd.DataFrame) -> Path:
    """
    Save extended CSV with "_with_invs" suffix.

    Args:
        original_path: Original CSV path
        df: Extended DataFrame to save

    Returns:
        Path to saved CSV
    """
    stem = original_path.stem
    suffix = original_path.suffix
    new_name = f"{stem}_with_invs{suffix}"
    new_path = original_path.parent / new_name

    df.to_csv(new_path, index=False)
    print(f"Saved extended CSV to: {new_path}")
    return new_path


def run_blender_generation(
    csv_path: Path,
    output_dir: Path,
    blender_path: str,
    blend_file: Path,
    fen_column: str = "FEN",
    img_name_column: str = "IMG_NAME",
) -> bool:
    """
    Run Blender to generate synthetic images from CSV.

    Args:
        csv_path: Path to CSV with FEN and IMG_NAME columns
        output_dir: Output directory for rendered images
        blender_path: Path to Blender executable
        blend_file: Path to chess-set.blend file
        fen_column: Column name for FEN strings
        img_name_column: Column name for image filenames

    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build Blender command
    blender_script = Path(__file__).parent.parent / "blender" / "generate_synthetic_from_fen.py"

    cmd = [
        blender_path,
        "-b", str(blend_file),
        "-P", str(blender_script),
        "--",
        "--csv", str(csv_path),
        "--output_dir", str(output_dir),
        "--fen_column", fen_column,
        "--img_name_column", img_name_column,
    ]

    print(f"Running Blender...")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        print("Blender output:")
        print(result.stdout)
        if result.stderr:
            print("Blender stderr:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Blender failed with exit code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: Blender not found at '{blender_path}'")
        print("Please specify the correct path with --blender_path")
        return False


def run_pipeline(
    csv_path: Path,
    output_dir: Path,
    blender_path: str,
    blend_file: Path,
    fen_column: str = "FEN",
    img_name_column: str = "IMG_NAME",
    skip_blender: bool = False,
    skip_crop: bool = False,
) -> bool:
    """
    Run the complete FEN inversion pipeline.

    Args:
        csv_path: Path to input CSV
        output_dir: Output directory for synthetic images
        blender_path: Path to Blender executable
        blend_file: Path to chess-set.blend file
        fen_column: Column name for FEN strings
        img_name_column: Column name for image filenames
        skip_blender: If True, skip Blender generation (just create CSV)
        skip_crop: If True, skip cropping step

    Returns:
        True if successful
    """
    print("=" * 60)
    print("Pipeline 1: FEN Inversion + Blender Generation")
    print("=" * 60)

    # Step 1: Load CSV
    print(f"\n[Step 1] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Validate columns
    if fen_column not in df.columns:
        print(f"  ERROR: FEN column '{fen_column}' not found")
        return False
    if img_name_column not in df.columns:
        print(f"  ERROR: IMG_NAME column '{img_name_column}' not found")
        return False

    # Step 2: Extend with inversions
    print(f"\n[Step 2] Creating inverted FEN duplicates...")
    extended_df = extend_csv_with_inversions(df, fen_column, img_name_column)
    print(f"  Extended to {len(extended_df)} rows ({len(df)} original + {len(df)} inverted)")

    # Step 3: Save extended CSV
    print(f"\n[Step 3] Saving extended CSV...")
    extended_csv_path = save_extended_csv(csv_path, extended_df)

    if skip_blender:
        print("\n[Step 4] Skipping Blender generation (--skip_blender)")
        return True

    # Step 4: Run Blender
    print(f"\n[Step 4] Running Blender generation...")
    print(f"  Output directory: {output_dir}")

    success = run_blender_generation(
        csv_path=extended_csv_path,
        output_dir=output_dir,
        blender_path=blender_path,
        blend_file=blend_file,
        fen_column=fen_column,
        img_name_column=img_name_column,
    )

    if not success:
        print("  ERROR: Blender generation failed")
        return False

    if skip_crop:
        print("\n[Step 5] Skipping crop step (--skip_crop)")
        return True

    # Step 5: Crop all images
    print(f"\n[Step 5] Cropping generated images...")
    crop_directory(str(output_dir), output_dir=None)  # Crop in-place

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="FEN Inversion + Blender Generation Pipeline"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with FEN and IMG_NAME columns"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for synthetic images"
    )
    parser.add_argument(
        "--blender_path",
        type=str,
        default="blender",
        help="Path to Blender executable (default: 'blender')"
    )
    parser.add_argument(
        "--blend_file",
        type=str,
        default="blender/chess-set.blend",
        help="Path to chess-set.blend file"
    )
    parser.add_argument(
        "--fen_column",
        type=str,
        default="FEN",
        help="Column name for FEN strings (default: 'FEN')"
    )
    parser.add_argument(
        "--img_name_column",
        type=str,
        default="IMG_NAME",
        help="Column name for image filenames (default: 'IMG_NAME')"
    )
    parser.add_argument(
        "--skip_blender",
        action="store_true",
        help="Skip Blender generation (just create extended CSV)"
    )
    parser.add_argument(
        "--skip_crop",
        action="store_true",
        help="Skip cropping step"
    )

    args = parser.parse_args()

    success = run_pipeline(
        csv_path=Path(args.csv),
        output_dir=Path(args.output),
        blender_path=args.blender_path,
        blend_file=Path(args.blend_file),
        fen_column=args.fen_column,
        img_name_column=args.img_name_column,
        skip_blender=args.skip_blender,
        skip_crop=args.skip_crop,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
