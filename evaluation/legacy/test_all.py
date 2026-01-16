"""
Comprehensive tests for all evaluation modules.
Run with: python test_all.py

Shows detailed input/output for each test.
"""

import sys
import numpy as np
import torch

np.set_printoptions(precision=2, suppress=True)


# ============================================
# Test 1: fen_to_grid.py
# ============================================
def test_fen_to_grid():
    print("\n" + "="*60)
    print("Testing: fen_to_grid.py")
    print("="*60)
    
    from fen_to_grid import fen_to_two_channel_grid, fen_to_two_channel_grid_numpy
    
    # Test 1.1: Starting position
    print("\n--- Test 1.1: Starting Position ---")
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"INPUT FEN: {start_fen}")
    
    grid = fen_to_two_channel_grid(start_fen, device='cpu')
    
    print(f"\nOUTPUT Shape: {grid.shape}")
    print("\nChannel 0 (WHITE pieces) - 1 means white piece present:")
    print(grid[0].numpy().astype(int))
    print("\nChannel 1 (BLACK pieces) - 1 means black piece present:")
    print(grid[1].numpy().astype(int))
    
    assert grid.shape == (2, 8, 8), f"Wrong shape: {grid.shape}"
    assert grid[1, 0, 0] == 1.0, "Black rook at a8 not detected"
    assert grid[0, 7, 0] == 1.0, "White rook at a1 not detected"
    print("\n✓ Starting position test PASSED")
    
    # Test 1.2: Empty board
    print("\n--- Test 1.2: Empty Board ---")
    empty_fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    print(f"INPUT FEN: {empty_fen}")
    
    grid_empty = fen_to_two_channel_grid(empty_fen, device='cpu')
    print(f"\nOUTPUT Sum of all values: {torch.sum(grid_empty).item()}")
    print("(Should be 0 - no pieces)")
    
    assert torch.sum(grid_empty) == 0.0, "Empty board should have no pieces"
    print("✓ Empty board test PASSED")
    
    # Test 1.3: Single piece
    print("\n--- Test 1.3: Single Piece (White King at d5) ---")
    single_fen = "8/8/8/3K4/8/8/8/8 w - - 0 1"
    print(f"INPUT FEN: {single_fen}")
    
    grid_single = fen_to_two_channel_grid(single_fen, device='cpu')
    print("\nOUTPUT Channel 0 (WHITE):")
    print(grid_single[0].numpy().astype(int))
    print(f"\nExpected: 1 at position [3,3] (row 3, col 3 = d5)")
    print(f"Actual value at [3,3]: {grid_single[0, 3, 3].item()}")
    
    assert grid_single[0, 3, 3] == 1.0, "White King at d5 not detected"
    print("✓ Single piece test PASSED")
    
    print("\n✓ All fen_to_grid tests PASSED!")
    return True


# ============================================
# Test 2: mask_to_grid.py
# ============================================
def test_mask_to_grid():
    print("\n" + "="*60)
    print("Testing: mask_to_grid.py")
    print("="*60)
    
    from mask_to_grid import masks_to_hard_grid
    
    H, W = 400, 400  # 8x8 grid -> 50x50 pixels per cell
    CELL = 50
    
    # Test 2.1: Full coverage detection
    print("\n--- Test 2.1: Full Cell Coverage ---")
    print(f"INPUT: Image size {H}x{W}, Cell size {CELL}x{CELL}")
    
    mask_white = np.zeros((H, W), dtype=np.float32)
    mask_white[0:CELL, 0:CELL] = 1.0  # Fill cell (0,0) completely
    print(f"White mask: Cell [0,0] filled with 100% white pixels")
    
    mask_black = np.zeros((H, W), dtype=np.float32)
    print(f"Black mask: Empty (all zeros)")
    
    grid = masks_to_hard_grid(mask_white, mask_black, threshold=0.20)
    
    print(f"\nOUTPUT Grid (threshold=20%):")
    print("Channel 0 (WHITE):")
    print(grid[0].astype(int))
    print(f"\nExpected: 1 at [0,0], 0 elsewhere")
    print(f"Actual value at [0,0]: {grid[0, 0, 0]}")
    
    assert grid[0, 0, 0] == 1.0, "Full white cell not detected"
    print("✓ Full coverage test PASSED")
    
    # Test 2.2: Threshold test
    print("\n--- Test 2.2: Partial Coverage (30%) ---")
    mask_partial = np.zeros((H, W), dtype=np.float32)
    # Fill 30% of cell (1,1)
    fill_size = int(np.sqrt(0.30 * CELL * CELL))
    mask_partial[CELL:CELL+fill_size, CELL:CELL+fill_size] = 1.0
    
    actual_coverage = (fill_size * fill_size) / (CELL * CELL) * 100
    print(f"INPUT: Cell [1,1] filled with ~{actual_coverage:.1f}% pixels")
    print(f"Threshold: 20%")
    
    grid_partial = masks_to_hard_grid(mask_partial, np.zeros_like(mask_partial), threshold=0.20)
    
    print(f"\nOUTPUT value at [1,1]: {grid_partial[0, 1, 1]}")
    print(f"Expected: 1 (because {actual_coverage:.1f}% > 20%)")
    
    assert grid_partial[0, 1, 1] == 1.0, "30% coverage should exceed 20% threshold"
    print("✓ Partial coverage test PASSED")
    
    # Test 2.3: Below threshold
    print("\n--- Test 2.3: Below Threshold (10%) ---")
    mask_small = np.zeros((H, W), dtype=np.float32)
    fill_small = int(np.sqrt(0.10 * CELL * CELL))
    mask_small[0:fill_small, 0:fill_small] = 1.0
    
    actual_small = (fill_small * fill_small) / (CELL * CELL) * 100
    print(f"INPUT: Cell [0,0] filled with ~{actual_small:.1f}% pixels")
    print(f"Threshold: 20%")
    
    grid_small = masks_to_hard_grid(mask_small, np.zeros_like(mask_small), threshold=0.20)
    
    print(f"\nOUTPUT value at [0,0]: {grid_small[0, 0, 0]}")
    print(f"Expected: 0 (because {actual_small:.1f}% < 20%)")
    
    assert grid_small[0, 0, 0] == 0.0, "10% coverage should NOT exceed 20% threshold"
    print("✓ Below threshold test PASSED")
    
    print("\n✓ All mask_to_grid tests PASSED!")
    return True


# ============================================
# Test 3: board_metrics.py
# ============================================
def test_board_metrics():
    print("\n" + "="*60)
    print("Testing: board_metrics.py")
    print("="*60)
    
    from board_metrics import (
        compute_board_accuracy,
        compute_cell_level_accuracy,
        compute_per_channel_accuracy
    )
    
    # Test 3.1: Perfect match
    print("\n--- Test 3.1: Perfect Match ---")
    gt = torch.zeros(2, 8, 8)
    gt[0, 0, 0] = 1.0  # White at (0,0)
    gt[1, 0, 4] = 1.0  # Black at (0,4)
    
    print("INPUT Ground Truth:")
    print(f"  White piece at [0,0]")
    print(f"  Black piece at [0,4]")
    
    pred_perfect = gt.clone()
    print("\nINPUT Prediction: (exact copy of GT)")
    
    acc = compute_board_accuracy(pred_perfect, gt)
    print(f"\nOUTPUT Accuracy: {acc:.2%}")
    print(f"Expected: 100%")
    
    assert acc == 1.0, f"Perfect match should be 100%, got {acc}"
    print("✓ Perfect match test PASSED")
    
    # Test 3.2: One cell wrong
    print("\n--- Test 3.2: One Cell Wrong ---")
    pred_wrong = gt.clone()
    pred_wrong[0, 0, 0] = 0.0  # Remove white piece (error!)
    
    print("INPUT Ground Truth: White at [0,0], Black at [0,4]")
    print("INPUT Prediction: Black at [0,4] only (missing white)")
    
    cell_acc = compute_cell_level_accuracy(pred_wrong, gt)
    expected = 63 / 64
    
    print(f"\nOUTPUT Cell Accuracy: {cell_acc:.2%}")
    print(f"Expected: {expected:.2%} (63/64 cells correct)")
    
    assert abs(cell_acc - expected) < 0.01, f"Expected ~{expected:.2%}, got {cell_acc:.2%}"
    print("✓ One cell wrong test PASSED")
    
    # Test 3.3: Per-channel accuracy
    print("\n--- Test 3.3: Per-Channel Accuracy ---")
    white_acc, black_acc = compute_per_channel_accuracy(pred_wrong, gt)
    
    print(f"\nOUTPUT:")
    print(f"  White channel accuracy: {white_acc:.2%}")
    print(f"  Black channel accuracy: {black_acc:.2%}")
    print(f"\nExpected: White < 100% (has error), Black = 100%")
    
    assert black_acc == 1.0, "Black channel should be perfect"
    assert white_acc < 1.0, "White channel should have error"
    print("✓ Per-channel test PASSED")
    
    print("\n✓ All board_metrics tests PASSED!")
    return True


# ============================================
# Test 4: Integration test
# ============================================
def test_integration():
    print("\n" + "="*60)
    print("Testing: Full Integration Pipeline")
    print("="*60)
    
    from fen_to_grid import fen_to_two_channel_grid_numpy
    from mask_to_grid import masks_to_hard_grid
    from board_metrics import compute_cell_level_accuracy
    
    print("\n--- Pipeline: FEN -> GT Grid -> Simulated Mask -> Pred Grid -> Accuracy ---")
    
    # Step 1: FEN to GT
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"\nStep 1 - INPUT FEN: {fen[:30]}...")
    
    gt_grid = fen_to_two_channel_grid_numpy(fen)
    print(f"Step 1 - OUTPUT GT Grid shape: {gt_grid.shape}")
    
    # Step 2: Simulate perfect masks
    print("\nStep 2 - Creating simulated masks that perfectly match FEN...")
    H, W = 400, 400
    CELL = 50
    
    mask_white = np.zeros((H, W), dtype=np.float32)
    mask_black = np.zeros((H, W), dtype=np.float32)
    
    # Fill masks based on GT grid
    for r in range(8):
        for c in range(8):
            if gt_grid[0, r, c] == 1.0:  # White piece
                mask_white[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL] = 1.0
            if gt_grid[1, r, c] == 1.0:  # Black piece
                mask_black[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL] = 1.0
    
    white_pixels = np.sum(mask_white > 0)
    black_pixels = np.sum(mask_black > 0)
    print(f"Step 2 - OUTPUT: {white_pixels} white pixels, {black_pixels} black pixels")
    
    # Step 3: Masks to Pred Grid
    print("\nStep 3 - Converting masks to predicted grid...")
    pred_grid = masks_to_hard_grid(mask_white, mask_black, threshold=0.20)
    print(f"Step 3 - OUTPUT Pred Grid shape: {pred_grid.shape}")
    
    # Step 4: Compare
    print("\nStep 4 - Comparing GT vs Pred...")
    gt_tensor = torch.from_numpy(gt_grid)
    pred_tensor = torch.from_numpy(pred_grid)
    
    acc = compute_cell_level_accuracy(pred_tensor, gt_tensor)
    
    print(f"\n{'='*40}")
    print(f"FINAL OUTPUT - Cell Accuracy: {acc:.2%}")
    print(f"{'='*40}")
    print(f"Expected: 100% (because masks perfectly match FEN)")
    
    assert acc == 1.0, f"Integration test should be 100%, got {acc}"
    print("\n✓ Integration test PASSED!")
    return True


# ============================================
# Main
# ============================================
def main():
    print("\n" + "#"*60)
    print("# RUNNING ALL EVALUATION TESTS (VERBOSE MODE)")
    print("#"*60)
    
    results = []
    
    try:
        results.append(("fen_to_grid", test_fen_to_grid()))
    except Exception as e:
        print(f"\n✗ fen_to_grid FAILED: {e}")
        results.append(("fen_to_grid", False))
    
    try:
        results.append(("mask_to_grid", test_mask_to_grid()))
    except Exception as e:
        print(f"\n✗ mask_to_grid FAILED: {e}")
        results.append(("mask_to_grid", False))
    
    try:
        results.append(("board_metrics", test_board_metrics()))
    except Exception as e:
        print(f"\n✗ board_metrics FAILED: {e}")
        results.append(("board_metrics", False))
    
    try:
        results.append(("integration", test_integration()))
    except Exception as e:
        print(f"\n✗ integration FAILED: {e}")
        results.append(("integration", False))
    
    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
