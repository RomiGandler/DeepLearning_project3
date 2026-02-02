"""
ETL module for data augmentation pipelines.

Provides three main pipelines:
1. FEN inversion + Blender generation (Pipeline 1)
2. SAM-based color inversion (Pipeline 2)
3. Geometric augmentations (Pipeline 3)

Usage:
    # Import specific modules as needed
    from etl.augmentations import fen_utils
    from etl.augmentations import geometric
    
    # Color swap requires ultralytics - import only when needed
    from etl.augmentations import color_swap
"""
