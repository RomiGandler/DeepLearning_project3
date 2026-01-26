"""
Chess board evaluation module.

Provides tools for evaluating chess board generation models by comparing
generated images against ground truth FEN positions using SAM-based detection.
"""

from src.evaluation.chess_eval_dataset import ChessEvalDataset
from src.evaluation.board_metrics import (
    compute_board_accuracy,
    compute_cell_level_accuracy,
    compute_per_channel_accuracy,
    compute_f1_metrics,
)
from src.evaluation.fen_to_grid import (
    fen_to_two_channel_grid,
    fen_to_two_channel_grid_numpy,
    batch_fen_to_grid,
)
from src.evaluation.evaluate_model import (
    evaluate_single,
    evaluate_batch,
    evaluate_folder,
)

__all__ = [
    # Dataset
    "ChessEvalDataset",
    # Metrics
    "compute_board_accuracy",
    "compute_cell_level_accuracy", 
    "compute_per_channel_accuracy",
    "compute_f1_metrics",
    # FEN conversion
    "fen_to_two_channel_grid",
    "fen_to_two_channel_grid_numpy",
    "batch_fen_to_grid",
    # Evaluation
    "evaluate_single",
    "evaluate_batch",
    "evaluate_folder",
]
