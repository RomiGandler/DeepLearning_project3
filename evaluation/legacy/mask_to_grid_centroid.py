"""
Module: Mask to 2-channel 8x8 grid conversion (Evaluation Mode).

Converts segmentation masks to hard board representation.
Uses a 'Winner-Takes-All' centroid-based logic:
Each detected piece (connected component) is assigned to the single grid cell 
containing its center of mass. This prevents one piece from triggering multiple cells.
"""

import numpy as np
import cv2

BOARD_SIZE = 8

def masks_to_hard_grid_centroid(
    mask_white: np.ndarray,
    mask_black: np.ndarray,
    threshold: float = 0.05,
    board_size: int = BOARD_SIZE,
    debug_path: str = None
) -> np.ndarray:
    """
    Convert boolean/float masks to (2, 8, 8) grid using 'Winner-Takes-All' centroid logic.
    Instead of checking coverage %, we find the center of each piece blob and assign it to 
    exactly one cell.
    
    Args:
        mask_white: (H, W) boolean or float array
        mask_black: (H, W) boolean or float array
        threshold: Minimum area (pixels) for a blob to be considered a piece (filter noise).
                   If threshold < 1.0, it is treated as a percentage of the cell area.
        board_size: Size of the board grid
        debug_path: Optional path to save a debug visualization image
    
    Returns:
        (2, 8, 8) numpy array with binary values (0.0 or 1.0)
    """
    h, w = mask_white.shape
    cell_h = h / board_size
    cell_w = w / board_size
    
    # Grid: Channel 0 = White, Channel 1 = Black
    grid = np.zeros((2, board_size, board_size), dtype=np.float32)

    # Prepare debug image if requested
    debug_img = None
    if debug_path:
        debug_img = np.zeros((h, w, 3), dtype=np.uint8)
        # Visualization: White=Red, Black=Blue
        debug_img[mask_white > 0] = [0, 0, 255] # Red
        debug_img[mask_black > 0] = [255, 0, 0] # Blue
        overlap = (mask_white > 0) & (mask_black > 0)
        debug_img[overlap] = [255, 0, 255] # Purple

    def process_channel(mask, channel_idx, color_bgr):
        # Ensure uint8 for connectedComponents
        if mask.dtype != np.uint8:
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
        else:
            mask_uint8 = mask

        # Find connected components (blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        # Iterate over components (skip label 0 which is background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Determine dynamic threshold
            # If threshold < 1.0, treat as fraction of theoretical cell area
            if threshold < 1.0:
                min_area = threshold * cell_h * cell_w
            else:
                min_area = threshold
            
            # Filter noise
            if area < min_area:
                continue

            # Get centroid of the blob
            cx, cy = centroids[i]
            
            # Determine which grid cell this centroid falls into
            r = int(cy / cell_h)
            c = int(cx / cell_w)
            
            # Assign to grid if within bounds
            if 0 <= r < board_size and 0 <= c < board_size:
                grid[channel_idx, r, c] = 1.0
                
                # Debug visualization: Draw centroid and label
                if debug_path:
                    # Draw centroid dot (Yellow)
                    cv2.circle(debug_img, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                    # Label cell with piece type (0=W, 1=B)
                    label = "W" if channel_idx == 0 else "B"
                    # text_color = (255, 255, 255) if channel_idx == 0 else (0, 255, 255)
                    # cv2.putText(debug_img, label, (int(cx)+5, int(cy)), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    # Process White Pieces (Channel 0)
    process_channel(mask_white, 0, (0, 0, 255))
    
    # Process Black Pieces (Channel 1)
    process_channel(mask_black, 1, (255, 0, 0))

    if debug_path:
        # Draw grid lines (Green)
        for r in range(board_size + 1):
            y = int(r * cell_h)
            cv2.line(debug_img, (0, y), (w, y), (0, 255, 0), 1)
        for c in range(board_size + 1):
            x = int(c * cell_w)
            cv2.line(debug_img, (x, 0), (x, h), (0, 255, 0), 1)
        
        cv2.imwrite(debug_path, debug_img)
    
    return grid

if __name__ == "__main__":
    print("Test run...")
    # Basic test
    h, w = 400, 400
    mw = np.zeros((h, w), dtype=np.uint8)
    mb = np.zeros((h, w), dtype=np.uint8)
    
    # Create a white blob at (50, 50) -> cell (1, 1) if size 8 and 400/8=50
    # 400/8 = 50px per cell. (50, 50) is exactly on corner of (0,0), (0,1), (1,0), (1,1).
    # Let's put it at 75, 75 -> Center of cell (1, 1)
    cv2.circle(mw, (75, 75), 15, 255, -1)
    
    grid = masks_to_hard_grid_centroid(mw, mb, threshold=0.1, debug_path="test_debug_centroid.png")
    print("Grid shape:", grid.shape)
    print("White at (1,1):", grid[0, 1, 1])
    assert grid[0, 1, 1] == 1.0
