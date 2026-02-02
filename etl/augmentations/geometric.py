"""
Geometric augmentation utilities for chess board images.

Creates all 8 distinct geometric transformations of a square (dihedral group D4):
1. identity - no transformation
2. rot90 - 90 degrees clockwise rotation
3. rot180 - 180 degrees rotation
4. rot270 - 270 degrees clockwise rotation
5. flip_h - horizontal flip (mirror over vertical axis)
6. flip_v - vertical flip (mirror over horizontal axis)
7. flip_diag - main diagonal flip (transpose)
8. flip_antidiag - anti-diagonal flip

For the augmentation pipeline, identity is excluded to avoid duplicates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image


# All D4 group transformations (including identity)
ALL_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda img: img,
    "rot90": lambda img: np.rot90(img, k=-1),      # 90 degrees clockwise
    "rot180": lambda img: np.rot90(img, k=2),      # 180 degrees
    "rot270": lambda img: np.rot90(img, k=1),      # 270 degrees clockwise (= 90 CCW)
    "flip_h": np.fliplr,                           # Horizontal flip
    "flip_v": np.flipud,                           # Vertical flip
    "flip_diag": lambda img: np.swapaxes(img, 0, 1),  # Main diagonal flip (transpose)
    "flip_antidiag": lambda img: np.rot90(np.swapaxes(img, 0, 1), k=2),  # Anti-diagonal
}

# Augmentation transforms (excluding identity to avoid duplicates)
AUGMENTATION_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    k: v for k, v in ALL_TRANSFORMS.items() if k != "identity"
}

# List of transform names for pipeline use
TRANSFORM_NAMES = list(AUGMENTATION_TRANSFORMS.keys())


def get_transforms(include_identity: bool = False) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """
    Get dictionary of geometric transformations.

    Args:
        include_identity: If True, include the identity (no-op) transform

    Returns:
        Dictionary mapping transform names to functions
    """
    if include_identity:
        return ALL_TRANSFORMS.copy()
    return AUGMENTATION_TRANSFORMS.copy()


def apply_transform(img: np.ndarray, transform_name: str) -> np.ndarray:
    """
    Apply a named geometric transformation to an image.

    Args:
        img: Input image as numpy array (H, W, C) or (H, W)
        transform_name: Name of transform (e.g., "rot90", "flip_h")

    Returns:
        Transformed image

    Raises:
        ValueError: If transform_name is not recognized
    """
    if transform_name not in ALL_TRANSFORMS:
        valid_names = list(ALL_TRANSFORMS.keys())
        raise ValueError(f"Unknown transform '{transform_name}'. Valid names: {valid_names}")

    return ALL_TRANSFORMS[transform_name](img)


def apply_all_transforms(img: np.ndarray, include_identity: bool = False) -> dict[str, np.ndarray]:
    """
    Apply all geometric transformations to an image.

    Args:
        img: Input image as numpy array (H, W, C) or (H, W)
        include_identity: If True, include identity transform in output

    Returns:
        Dictionary mapping transform names to transformed images
    """
    transforms = get_transforms(include_identity=include_identity)
    return {name: func(img) for name, func in transforms.items()}


def has_transform_suffix(filename: str) -> bool:
    """
    Check if a filename already has a geometric transform suffix.

    Args:
        filename: Image filename to check

    Returns:
        True if filename ends with a transform suffix (e.g., "_rot90")
    """
    stem = Path(filename).stem
    for name in ALL_TRANSFORMS.keys():
        if stem.endswith(f"_{name}"):
            return True
    return False


def add_transform_suffix(filename: str, transform_name: str) -> str:
    """
    Add a transform suffix to a filename before the extension.

    Args:
        filename: Original filename (e.g., "board.png")
        transform_name: Transform name (e.g., "rot90")

    Returns:
        Filename with suffix (e.g., "board_rot90.png")
    """
    path = Path(filename)
    return f"{path.stem}_{transform_name}{path.suffix}"


def process_image_file(
    image_path: Path,
    output_dir: Path,
    include_identity: bool = False,
) -> list[Path]:
    """
    Process a single image file and save all geometric augmentations.

    Args:
        image_path: Path to input image
        output_dir: Directory to save augmented images
        include_identity: If True, also save identity (unchanged) image

    Returns:
        List of paths to saved augmented images
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output format
    output_ext = image_path.suffix

    # Generate and save augmentations
    saved_paths = []
    transforms = get_transforms(include_identity=include_identity)

    for name, transform_func in transforms.items():
        aug_array = transform_func(img_array)
        aug_img = Image.fromarray(aug_array)

        output_name = add_transform_suffix(image_path.name, name)
        output_path = output_dir / output_name

        aug_img.save(output_path)
        saved_paths.append(output_path)

    return saved_paths


def augment_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    include_identity: bool = False,
    skip_augmented: bool = True,
) -> list[Path]:
    """
    Apply geometric augmentations to all images in a directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Output directory (defaults to input_dir for in-place)
        include_identity: If True, also create identity copies
        skip_augmented: If True, skip images that already have transform suffixes

    Returns:
        List of all saved augmented image paths
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    # Filter out already-augmented images if requested
    if skip_augmented:
        image_files = [f for f in image_files if not has_transform_suffix(f.name)]

    print(f"Found {len(image_files)} images to augment")

    all_saved = []
    for i, img_path in enumerate(image_files, 1):
        print(f"  [{i}/{len(image_files)}] Processing {img_path.name}")
        saved = process_image_file(img_path, output_dir, include_identity=include_identity)
        all_saved.extend(saved)

    print(f"Saved {len(all_saved)} augmented images to {output_dir}")
    return all_saved
