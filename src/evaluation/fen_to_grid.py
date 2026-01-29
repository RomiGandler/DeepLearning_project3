"""
Module 1: FEN to 2-channel 8x8 grid conversion.

Converts FEN notation to a differentiable-friendly 2-channel representation:
- Channel 0: White pieces (1.0 where white, 0.0 elsewhere)
- Channel 1: Black pieces (1.0 where black, 0.0 elsewhere)
"""

import torch
import numpy as np

BOARD_SIZE = 8


def fen_to_two_channel_grid(fen: str, device: str = 'cuda') -> torch.Tensor:
    """
    Parses a FEN string into a 2-channel 8x8 grid.
    
    Args:
        fen: FEN string (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        device: Device to place the tensor on ('cuda' or 'cpu')
    
    Returns:
        Tensor of shape (2, 8, 8):
        - Channel 0: White pieces (1.0 where white piece exists, 0.0 elsewhere)
        - Channel 1: Black pieces (1.0 where black piece exists, 0.0 elsewhere)
    """
    grid = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=device)
    
    # FEN placement is the first part of the string
    placement = fen.split(' ')[0]
    rows = placement.split('/')
    
    for r_idx, row in enumerate(rows):
        c_idx = 0
        for char in row:
            if char.isdigit():
                # Empty squares
                c_idx += int(char)
            else:
                # Piece: uppercase = white, lowercase = black
                if char.isupper():
                    grid[0, r_idx, c_idx] = 1.0  # White channel
                else:
                    grid[1, r_idx, c_idx] = 1.0  # Black channel
                c_idx += 1
    
    return grid


def fen_to_two_channel_grid_numpy(fen: str) -> np.ndarray:
    """
    Numpy version for non-differentiable use cases.
    
    Args:
        fen: FEN string
    
    Returns:
        ndarray of shape (2, 8, 8) with float32 values
    """
    grid = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    placement = fen.split(' ')[0]
    rows = placement.split('/')
    
    for r_idx, row in enumerate(rows):
        c_idx = 0
        for char in row:
            if char.isdigit():
                c_idx += int(char)
            else:
                if char.isupper():
                    grid[0, r_idx, c_idx] = 1.0
                else:
                    grid[1, r_idx, c_idx] = 1.0
                c_idx += 1
    
    return grid


def batch_fen_to_grid(fens: list, device: str = 'cuda') -> torch.Tensor:
    """
    Convert a batch of FEN strings to grids.
    
    Args:
        fens: List of FEN strings
        device: Device to place the tensor on
    
    Returns:
        Tensor of shape (B, 2, 8, 8)
    """
    grids = [fen_to_two_channel_grid(fen, device) for fen in fens]
    return torch.stack(grids, dim=0)


if __name__ == "__main__":
    print("Running comprehensive tests for fen_to_grid...")
    
    # Test Case 1: Starting Position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    grid = fen_to_two_channel_grid(start_fen, device='cpu')
    
    assert grid.shape == (2, 8, 8)
    assert grid[1, 0, 0] == 1.0  # Black rook (a8)
    assert grid[1, 0, 4] == 1.0  # Black king (e8)
    assert grid[0, 7, 3] == 1.0  # White queen (d1)
    assert grid[0, 4, 4] == 0.0  # Empty square (e4)
    print("✓ Starting position test passed")

    # Test Case 2: Empty Board
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    grid_empty = fen_to_two_channel_grid(empty_fen, device='cpu')
    assert torch.sum(grid_empty) == 0.0
    print("✓ Empty board test passed")

    # Test Case 3: Specific Piece Placement (Middle of board)
    # White King at e4, Black Queen at d5
    mid_fen = "8/8/8/3q4/4K3/8/8/8 w - - 0 1"
    grid_mid = fen_to_two_channel_grid(mid_fen, device='cpu')
    assert grid_mid[0, 4, 4] == 1.0  # White King (e4 -> row 4, col 4)
    assert grid_mid[1, 3, 3] == 1.0  # Black Queen (d5 -> row 3, col 3)
    print("✓ Specific placement test passed")

    print("All fen_to_grid tests passed!")
