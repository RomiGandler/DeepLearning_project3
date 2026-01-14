"""
Module: Mask to 2-channel 8x8 grid conversion (Evaluation Mode).

Converts segmentation masks to hard board representation using thresholding.
"""

import numpy as np

BOARD_SIZE = 8

def masks_to_hard_grid(
    mask_white: np.ndarray,
    mask_black: np.ndarray,
    threshold: float = 0.20,
    board_size: int = BOARD_SIZE
) -> np.ndarray:
    """
    Convert boolean/float masks to (2, 8, 8) grid using coverage thresholding.
    
    Args:
        mask_white: (H, W) boolean or float array
        mask_black: (H, W) boolean or float array
        threshold: Coverage threshold to consider a cell occupied
        board_size: Size of the board grid
    
    Returns:
        (2, 8, 8) numpy array with binary values (0.0 or 1.0)
    """
    h, w = mask_white.shape
    cell_h = h / board_size
    cell_w = w / board_size
    
    grid = np.zeros((2, board_size, board_size), dtype=np.float32)
    
    for r in range(board_size):
        for c in range(board_size):
            # Define cell coordinates
            y_start = int(r * cell_h)
            y_end = int((r + 1) * cell_h)
            x_start = int(c * cell_w)
            x_end = int((c + 1) * cell_w)
            
            # Extract cell crops
            white_cell = mask_white[y_start:y_end, x_start:x_end]
            black_cell = mask_black[y_start:y_end, x_start:x_end]
            
            cell_area = (y_end - y_start) * (x_end - x_start)
            if cell_area == 0:
                continue
            
            white_coverage = np.sum(white_cell) / cell_area
            black_coverage = np.sum(black_cell) / cell_area
            
            # Apply threshold - prioritize higher coverage
            if white_coverage > threshold and white_coverage >= black_coverage:
                grid[0, r, c] = 1.0
            elif black_coverage > threshold and black_coverage > white_coverage:
                grid[1, r, c] = 1.0
    
    return grid

if __name__ == "__main__":
    print("Running comprehensive tests for mask_to_grid...")
    
    # Setup: 200x200 image, 8x8 grid -> 25x25 pixels per cell
    H, W = 200, 200
    BOARD_SIZE = 8
    
    # 1. Test Perfect Coverage (Hard Grid)
    mask_white = np.zeros((H, W), dtype=np.float32)
    # Fill cell (0,0) completely
    mask_white[0:25, 0:25] = 1.0
    
    mask_black = np.zeros((H, W), dtype=np.float32)
    # Fill cell (1,1) completely
    mask_black[25:50, 25:50] = 1.0
    
    hard_grid = masks_to_hard_grid(mask_white, mask_black, board_size=BOARD_SIZE)
    assert hard_grid[0, 0, 0] == 1.0
    assert hard_grid[1, 1, 1] == 1.0
    assert hard_grid[0, 0, 1] == 0.0 # Should be empty
    print("✓ Hard grid perfect coverage test passed")

    # 2. Test Partial Coverage (Thresholding)
    mask_partial = np.zeros((H, W), dtype=np.float32)
    # Fill 30% of cell (2,2) -> should pass 0.20 threshold
    # 30% of 25x25 = 187.5 pixels. sqrt(187.5) approx 13.7
    fill_h = 14
    mask_partial[50:50+fill_h, 50:50+fill_h] = 1.0
    
    hard_grid_partial = masks_to_hard_grid(mask_partial, np.zeros_like(mask_partial), threshold=0.20)
    assert hard_grid_partial[0, 2, 2] == 1.0
    print("✓ Hard grid partial coverage test passed")

    print("All mask_to_grid tests passed!")
