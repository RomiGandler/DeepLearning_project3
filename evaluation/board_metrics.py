"""
Module: Evaluation metrics for chess board generation.

Provides accuracy metrics for comparing predicted and ground truth boards.
"""

import numpy as np
import torch
from typing import Tuple

def compute_board_accuracy(
    pred_grid: torch.Tensor,
    gt_grid: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute accuracy (element-wise match rate).
    """
    with torch.no_grad():
        if pred_grid.dim() == 3:
            pred_grid = pred_grid.unsqueeze(0)
        if gt_grid.dim() == 3:
            gt_grid = gt_grid.unsqueeze(0)
        
        # Convert soft predictions to hard
        pred_hard = (pred_grid > threshold).float()
        
        # Compare element-wise
        correct = (pred_hard == gt_grid).float()
        accuracy = correct.mean().item()
    
    return accuracy

def compute_cell_level_accuracy(
    pred_grid: torch.Tensor,
    gt_grid: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute cell-level accuracy (both channels must match for a cell).
    """
    with torch.no_grad():
        if pred_grid.dim() == 3:
            pred_grid = pred_grid.unsqueeze(0)
        if gt_grid.dim() == 3:
            gt_grid = gt_grid.unsqueeze(0)
        
        pred_hard = (pred_grid > threshold).float()
        
        white_match = pred_hard[:, 0] == gt_grid[:, 0]
        black_match = pred_hard[:, 1] == gt_grid[:, 1]
        
        cell_correct = (white_match & black_match).float()
        accuracy = cell_correct.mean().item()
    
    return accuracy

def compute_per_channel_accuracy(
    pred_grid: torch.Tensor,
    gt_grid: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute accuracy per channel (white and black separately).
    """
    with torch.no_grad():
        if pred_grid.dim() == 3:
            pred_grid = pred_grid.unsqueeze(0)
        if gt_grid.dim() == 3:
            gt_grid = gt_grid.unsqueeze(0)
        
        pred_hard = (pred_grid > threshold).float()
        
        white_correct = (pred_hard[:, 0] == gt_grid[:, 0]).float()
        white_acc = white_correct.mean().item()
        
        black_correct = (pred_hard[:, 1] == gt_grid[:, 1]).float()
        black_acc = black_correct.mean().item()
    
    return white_acc, black_acc
