"""
Evaluation script for chess board generation model.

Supports three evaluation modes:
- evaluate_single: Single pre-loaded image
- evaluate_batch: Batch of pre-loaded images  
- evaluate_folder: Load from directory using FenDataLoader

Usage:
    python evaluate_model.py --generated_dir path/to/outputs --csv_path path/to/data.csv
"""

import os
import argparse
import numpy as np
import torch
from typing import Optional, Dict, List

from fen_data_loader import FenDataLoader
from data_saver import DataSaver
from sam_grid_extractor_with_centroids import SAMGridExtractor
from fen_to_grid import fen_to_two_channel_grid_numpy
from board_metrics import (
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
        extractor: SAMGridExtractor instance
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
        extractor: SAMGridExtractor instance
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
    generated_dir: str,
    csv_path: str,
    output_dir: Optional[str] = None,
    gt_images_dir: Optional[str] = None,
    model_path: str = "/home/avinoamd/roni/BBDM/SAM/sam3.pt",
    save_debug: bool = True,
) -> Dict:
    """
    Evaluate all images in a folder using FenDataLoader.
    
    Args:
        generated_dir: Directory containing generated images
        csv_path: Path to CSV with FEN labels
        output_dir: Output directory for results
        gt_images_dir: Optional directory with ground truth images
        model_path: Path to SAM model
        save_debug: Whether to save debug outputs
        
    Returns:
        Summary dict with aggregated metrics
    """
    print(f"Generated dir: {generated_dir}")
    print(f"CSV path: {csv_path}")
    
    loader = FenDataLoader(csv_path, generated_dir)
    extractor = SAMGridExtractor(model_path=model_path)
    
    saver = None
    if save_debug:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(generated_dir), "eval_output")
        saver = DataSaver(output_dir, gt_images_dir)
    
    # Collect all data
    images, fens, file_ids = [], [], []
    for image_path, fen, image in loader:
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
    parser.add_argument("--generated_dir", type=str, 
                        default="/home/avinoamd/roni/evaluation/images_to_eval/model_output")
    parser.add_argument("--csv_path", type=str,
                        default="/home/avinoamd/roni/BBDM/data_10.01_no_hands/val/val_data.csv")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gt_images_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="/home/avinoamd/roni/BBDM/SAM/sam3.pt")
    parser.add_argument("--no_debug", action="store_true")
    
    args = parser.parse_args()
    
    evaluate_folder(
        generated_dir=args.generated_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        gt_images_dir=args.gt_images_dir,
        model_path=args.model_path,
        save_debug=not args.no_debug,
    )


if __name__ == "__main__":
    main()
