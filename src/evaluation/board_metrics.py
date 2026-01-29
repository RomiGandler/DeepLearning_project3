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

def compute_f1_metrics(
    pred_grid: torch.Tensor,
    gt_grid: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7
) -> dict:
    """
    Compute Precision, Recall, and F1 Score for White and Black channels.
    """
    with torch.no_grad():
        if pred_grid.dim() == 3:
            pred_grid = pred_grid.unsqueeze(0)
        if gt_grid.dim() == 3:
            gt_grid = gt_grid.unsqueeze(0)
        
        pred_hard = (pred_grid > threshold).float()
        
        metrics = {}
        channels = ['white', 'black']
        
        for i, color in enumerate(channels):
            # True Positives: Predicted 1 AND Actual 1
            tp = ((pred_hard[:, i] == 1) & (gt_grid[:, i] == 1)).float().sum()
            
            # False Positives: Predicted 1 but Actual 0 (Hallucination)
            fp = ((pred_hard[:, i] == 1) & (gt_grid[:, i] == 0)).float().sum()
            
            # False Negatives: Predicted 0 but Actual 1 (Missed Piece)
            fn = ((pred_hard[:, i] == 0) & (gt_grid[:, i] == 1)).float().sum()
            
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            
            metrics[f'{color}_precision'] = precision.item()
            metrics[f'{color}_recall'] = recall.item()
            metrics[f'{color}_f1'] = f1.item()
            
        # Average F1
        metrics['avg_f1'] = (metrics['white_f1'] + metrics['black_f1']) / 2
        
        return metrics
