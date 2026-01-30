"""
Augmentation utilities for chess images.

Modules:
- fen_utils: FEN string manipulation (color inversion)
- color_swap: SAM-based piece color swapping (requires ultralytics)
- geometric: D4 group geometric transformations

Usage:
    # Lightweight imports (no heavy dependencies)
    from etl.augmentations.fen_utils import invert_fen_colors
    from etl.augmentations.geometric import get_transforms, apply_transform
    
    # SAM-based color swap (requires ultralytics + torch)
    from etl.augmentations.color_swap import SAMColorSwapper
"""

# Only expose lightweight utilities by default
from etl.augmentations.fen_utils import invert_fen_colors, add_inv_suffix, has_inv_suffix
from etl.augmentations.geometric import get_transforms, apply_transform, TRANSFORM_NAMES

__all__ = [
    "invert_fen_colors",
    "add_inv_suffix",
    "has_inv_suffix",
    "get_transforms",
    "apply_transform",
    "TRANSFORM_NAMES",
]
