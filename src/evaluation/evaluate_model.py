"""
Evaluation script for chess board generation model.

Supports three evaluation modes:
- evaluate_single: Single pre-loaded image
- evaluate_batch: Batch of pre-loaded images  
- evaluate_folder: Load from directory using ChessEvalDataset

Usage:
    # Evaluate model outputs against test set
    python -m src.evaluation.evaluate_model --dataset_path path/to/data --stage test --generated_dir path/to/outputs
    
    # Evaluate original B/ images (sanity check)
    python -m src.evaluation.evaluate_model --dataset_path path/to/data --stage test
    
    # Auto-download dataset from HuggingFace
    python -m src.evaluation.evaluate_model --stage test --generated_dir path/to/outputs
"""

import os
import argparse
import numpy as np
import torch
from typing import Optional, Dict, List

from src.evaluation.dataloader import ChessEvalDataset
from src.evaluation.data_saver import DataSaver
from src.evaluation.sam_grid_extractor import SAMGridExtractor
from src.evaluation.fen_to_grid import fen_to_two_channel_grid_numpy
from src.evaluation.board_metrics import (
    compute_board_accuracy,
    compute_cell_level_accuracy,
    compute_per_channel_accuracy,
    compute_f1_metrics
)


def compute_metrics(pred_grid: np.ndarray, gt_grid: np.ndarray) -> Dict:
    """Compute all metrics between predicted and ground truth grids."""
    gt_tensor = torch.from_numpy(gt_grid)
    pred_tensor = torch.from_numpy(pred_grid)
    
    return {
        'overall_accuracy': compute_board_accuracy(pred_tensor, gt_tensor),
        'cell_accuracy': compute_cell_level_accuracy(pred_tensor, gt_tensor),
        'white_accuracy': compute_per_channel_accuracy(pred_tensor, gt_tensor)[0],
        'black_accuracy': compute_per_channel_accuracy(pred_tensor, gt_tensor)[1],
        **compute_f1_metrics(pred_tensor, gt_tensor),
    }


def evaluate_single(
    image: np.ndarray,
    fen: str,
    file_id: str,
    extractor: SAMGridExtractor,
    saver: Optional[DataSaver] = None,
) -> Dict:
    """
    Evaluate a single pre-loaded image.
    
    Args:
        image: BGR image as numpy array
        fen: FEN string for ground truth
        file_id: Identifier for this sample
        extractor: GridExtractor instance
        saver: Optional DataSaver for debug outputs
        
    Returns:
        Dict with metrics and grids
    """
    gt_grid = fen_to_two_channel_grid_numpy(fen)
    pred_grid, detections = extractor.extract_grid(image)
    
    if saver:
        saver.save_input_image(file_id, image)
        saver.save_gt_image(file_id)
        saver.save_grid(file_id, gt_grid, "grid_gt")
        saver.save_grid(file_id, pred_grid, "grid_pred")
        saver.save_detections_debug(file_id, image, detections, pred_grid)
    
    metrics = compute_metrics(pred_grid, gt_grid)
    metrics['gt_grid'] = gt_grid
    metrics['pred_grid'] = pred_grid
    metrics['detections'] = detections
    
    return metrics


def evaluate_batch(
    images: List[np.ndarray],
    fens: List[str],
    file_ids: List[str],
    extractor: SAMGridExtractor,
    saver: Optional[DataSaver] = None,
) -> List[Dict]:
    """
    Evaluate a batch of pre-loaded images.
    
    Args:
        images: List of BGR images
        fens: List of FEN strings
        file_ids: List of identifiers
        extractor: GridExtractor instance
        saver: Optional DataSaver for debug outputs
        
    Returns:
        List of metric dicts
    """
    assert len(images) == len(fens) == len(file_ids), "Mismatched input lengths"
    
    results = []
    for image, fen, file_id in zip(images, fens, file_ids):
        metrics = evaluate_single(image, fen, file_id, extractor, saver)
        results.append(metrics)
        print(f"  {file_id}: Cell Acc={metrics['cell_accuracy']:.2%}, "
              f"W-F1={metrics['white_f1']:.2%} (R={metrics['white_recall']:.2%}), "
              f"B-F1={metrics['black_f1']:.2%} (R={metrics['black_recall']:.2%})")
    
    return results


def evaluate_folder(
    dataset_path: Optional[str] = None,
    stage: str = 'test',
    generated_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    gt_images_dir: Optional[str] = None,
    save_debug: bool = True,
) -> Dict:
    """
    Evaluate all images using ChessEvalDataset.
    
    The SAM model (sam3.pt) is automatically downloaded from HuggingFace
    if not found in the local checkpoints directory.
    
    Args:
        dataset_path: Root dataset path. If None, downloads from HuggingFace.
        stage: Which split to evaluate ('train', 'val', 'test')
        generated_dir: Model output directory. If None, evaluates original B/ images.
        output_dir: Output directory for results
        gt_images_dir: Optional directory with ground truth images for comparison
        save_debug: Whether to save debug outputs
        
    Returns:
        Summary dict with aggregated metrics
    """
    print(f"Dataset path: {dataset_path}")
    print(f"Stage: {stage}")
    print(f"Generated dir: {generated_dir}")
    
    # Create unified dataset
    dataset = ChessEvalDataset(
        dataset_path=dataset_path,
        stage=stage,
        generated_dir=generated_dir,
    )
    
    # Create extractor (auto-downloads SAM model from HuggingFace if needed)
    extractor = SAMGridExtractor()
    
    # Setup saver
    saver = None
    if save_debug:
        if output_dir is None:
            if generated_dir:
                output_dir = os.path.join(os.path.dirname(generated_dir), "eval_output")
            else:
                output_dir = "eval_output"
        saver = DataSaver(output_dir, gt_images_dir)
    
    # Collect all data from dataset
    images, fens, file_ids = [], [], []
    for image_path, fen, image in dataset:
        file_id = os.path.splitext(os.path.basename(image_path))[0]
        images.append(image)
        fens.append(fen)
        file_ids.append(file_id)
    
    # Evaluate batch
    results = evaluate_batch(images, fens, file_ids, extractor, saver)
    
    assert results, "No results computed"
    
    summary = {
        'num_evaluated': len(results),
        'avg_overall_accuracy': float(np.mean([r['overall_accuracy'] for r in results])),
        'avg_cell_accuracy': float(np.mean([r['cell_accuracy'] for r in results])),
        'avg_white_f1': float(np.mean([r['white_f1'] for r in results])),
        'avg_black_f1': float(np.mean([r['black_f1'] for r in results])),
        'avg_white_recall': float(np.mean([r['white_recall'] for r in results])),
        'avg_black_recall': float(np.mean([r['black_recall'] for r in results])),
        'avg_white_precision': float(np.mean([r['white_precision'] for r in results])),
        'avg_black_precision': float(np.mean([r['black_precision'] for r in results])),
    }
    summary['error_rate'] = 1 - summary['avg_cell_accuracy']
    
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Images Evaluated: {summary['num_evaluated']}")
    print(f"Average Cell Accuracy: {summary['avg_cell_accuracy']:.2%}")
    print(f"Average White - F1: {summary['avg_white_f1']:.2%}, Precision: {summary['avg_white_precision']:.2%}, Recall: {summary['avg_white_recall']:.2%}")
    print(f"Average Black - F1: {summary['avg_black_f1']:.2%}, Precision: {summary['avg_black_precision']:.2%}, Recall: {summary['avg_black_recall']:.2%}")
    print(f"Error Rate: {summary['error_rate']:.2%}")
    
    if saver:
        saver.save_summary(summary)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess board generation model")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Root dataset path. If not provided, downloads from HuggingFace.")
    parser.add_argument("--stage", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate")
    parser.add_argument("--generated_dir", type=str, default=None,
                        help="Path to model outputs. If not provided, evaluates original B/ images.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--gt_images_dir", type=str, default=None,
                        help="Optional directory with ground truth images for comparison")
    parser.add_argument("--no_debug", action="store_true",
                        help="Disable debug output saving")
    
    args = parser.parse_args()
    
    evaluate_folder(
        dataset_path=args.dataset_path,
        stage=args.stage,
        generated_dir=args.generated_dir,
        output_dir=args.output_dir,
        gt_images_dir=args.gt_images_dir,
        save_debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()
