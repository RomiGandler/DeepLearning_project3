"""
Module 5: Main evaluation script.

Evaluates model outputs by:
1. Loading FEN from CSV
2. Converting FEN to ground truth grid
3. Running SAM on generated images to get masks
4. Converting masks to predicted grid
5. Computing accuracy/metrics

Usage:
    python evaluate_model.py --generated_dir path/to/outputs --csv_path path/to/data.csv
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from typing import Optional, Dict

# Import the modules
from fen_to_grid import fen_to_two_channel_grid_numpy
from sam_mask_extractor import SAMMaskExtractor
from mask_to_grid import masks_to_hard_grid
from board_metrics import (
    compute_board_accuracy,
    compute_cell_level_accuracy,
    compute_per_channel_accuracy
)


def get_file_id(filename: str) -> str:
    """Extract numeric ID from filename."""
    nums = re.findall(r'\d+', filename)
    if nums:
        return nums[-1]
    return filename


def load_csv_fen_mapping(csv_path: str) -> Dict[str, str]:
    """Load FEN mapping from CSV file."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    fen_col = 'FEN' if 'FEN' in df.columns else df.columns[0]
    filename_col = None
    for col in df.columns:
        if df[col].astype(str).str.contains('.png').any():
            filename_col = col
            break
    
    if filename_col is None:
        raise ValueError("Could not find filename column in CSV")
    
    fen_map = {}
    for _, row in df.iterrows():
        fname = str(row[filename_col])
        fid = get_file_id(fname)
        fen_map[fid] = row[fen_col]
    
    return fen_map


def save_grid_as_image(grid: np.ndarray, save_path: str, cell_size: int = 50):
    """
    Visualize 2-channel grid as an RGB image.
    White pieces = Green, Black pieces = Red, Overlap = Yellow.
    """
    board_size = grid.shape[1]
    h, w = board_size * cell_size, board_size * cell_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # White channel -> Green (0, 255, 0)
    # Black channel -> Red (0, 0, 255) in BGR
    
    for r in range(board_size):
        for c in range(board_size):
            y1, y2 = r * cell_size, (r + 1) * cell_size
            x1, x2 = c * cell_size, (c + 1) * cell_size
            
            is_white = grid[0, r, c] > 0.5
            is_black = grid[1, r, c] > 0.5
            
            color = (0, 0, 0)
            if is_white and is_black:
                color = (0, 255, 255) # Yellow (Error state usually)
            elif is_white:
                color = (0, 255, 0) # Green
            elif is_black:
                color = (0, 0, 255) # Red
                
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            # Draw border
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 1)
            
    cv2.imwrite(save_path, img)


def evaluate_single_image(
    image_path: str,
    fen: str,
    sam_extractor: SAMMaskExtractor,
    threshold: float = 0.20,
    debug_dir: Optional[str] = None,
    file_id: Optional[str] = None
) -> Dict:
    """
    Evaluate a single image against its FEN ground truth.
    """
    # 1. FEN to ground truth grid (NumPy)
    gt_grid = fen_to_two_channel_grid_numpy(fen)
    
    # 2. Extract masks using SAM (NumPy)
    mask_white, mask_black = sam_extractor.extract_masks(image_path)
    
    if debug_dir and file_id:
        sam_extractor.save_debug_masks(mask_white, mask_black, debug_dir, file_id)
        # Save GT as visual image
        save_grid_as_image(gt_grid, os.path.join(debug_dir, f"{file_id}_gt_visual.png"))
        # Save GT as text
        np.savetxt(os.path.join(debug_dir, f"{file_id}_gt_grid.txt"), 
                   gt_grid[0] + gt_grid[1] * 2, fmt='%d')
    
    # 3. Convert masks to grid (Hard 0/1 Grid)
    pred_grid = masks_to_hard_grid(mask_white, mask_black, threshold=threshold)
    
    if debug_dir and file_id:
        # Save Pred as visual image
        save_grid_as_image(pred_grid, os.path.join(debug_dir, f"{file_id}_pred_visual.png"))
        # Save Pred as text
        np.savetxt(os.path.join(debug_dir, f"{file_id}_pred_grid.txt"),
                   pred_grid[0] + pred_grid[1] * 2, fmt='%d')
    
    # 4. Compute metrics
    gt_tensor = torch.from_numpy(gt_grid)
    pred_tensor = torch.from_numpy(pred_grid)
    
    overall_acc = compute_board_accuracy(pred_tensor, gt_tensor)
    cell_acc = compute_cell_level_accuracy(pred_tensor, gt_tensor)
    white_acc, black_acc = compute_per_channel_accuracy(pred_tensor, gt_tensor)
    
    return {
        'overall_accuracy': overall_acc,
        'cell_accuracy': cell_acc,
        'white_accuracy': white_acc,
        'black_accuracy': black_acc,
        'gt_grid': gt_grid,
        'pred_grid': pred_grid
    }


def evaluate_folder(
    generated_dir: str,
    csv_path: str,
    model_path: str = "/home/avinoamd/roni/BBDM/SAM/sam3.pt",
    threshold: float = 0.20,
    save_debug: bool = True
) -> Dict:
    """Evaluate all images in a folder."""
    print(f"Starting evaluation...")
    print(f"Generated dir: {generated_dir}")
    print(f"CSV path: {csv_path}")
    
    # #region agent log
    import json; open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"A","location":"evaluate_model.py:164","message":"generated_dir exists check","data":{"generated_dir":generated_dir,"exists":os.path.exists(generated_dir),"isdir":os.path.isdir(generated_dir) if os.path.exists(generated_dir) else False},"timestamp":__import__('time').time()})+'\n')
    # #endregion
    
    # #region agent log
    all_files_in_dir = os.listdir(generated_dir) if os.path.exists(generated_dir) else []
    open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"B","location":"evaluate_model.py:168","message":"all files in dir","data":{"all_files":all_files_in_dir[:20],"total_count":len(all_files_in_dir)},"timestamp":__import__('time').time()})+'\n')
    # #endregion
    
    fen_map = load_csv_fen_mapping(csv_path)
    print(f"Loaded {len(fen_map)} FEN entries from CSV")
    
    # #region agent log
    open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"D","location":"evaluate_model.py:175","message":"fen_map loaded","data":{"fen_map_size":len(fen_map),"sample_keys":list(fen_map.keys())[:10]},"timestamp":__import__('time').time()})+'\n')
    # #endregion
    
    sam_extractor = SAMMaskExtractor(model_path=model_path)
    if not sam_extractor.is_available():
        raise RuntimeError("SAM extractor not available")
    
    gen_files = [f for f in os.listdir(generated_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(gen_files)} images in generated dir")
    
    # #region agent log
    open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"B","location":"evaluate_model.py:186","message":"filtered gen_files","data":{"gen_files":gen_files[:10],"count":len(gen_files)},"timestamp":__import__('time').time()})+'\n')
    # #endregion
    
    # #region agent log
    sample_ids = [get_file_id(f) for f in gen_files[:10]]
    open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"C","location":"evaluate_model.py:191","message":"extracted IDs from gen_files","data":{"sample_filenames":gen_files[:10],"sample_ids":sample_ids},"timestamp":__import__('time').time()})+'\n')
    # #endregion
    
    # #region agent log
    matching_ids = [fid for fid in sample_ids if fid in fen_map]
    non_matching_ids = [fid for fid in sample_ids if fid not in fen_map]
    open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"E","location":"evaluate_model.py:197","message":"ID matching check","data":{"matching_ids":matching_ids,"non_matching_ids":non_matching_ids,"fen_map_sample_keys":list(fen_map.keys())[:5]},"timestamp":__import__('time').time()})+'\n')
    # #endregion
    
    debug_dir = None
    if save_debug:
        debug_dir = os.path.join(os.path.dirname(generated_dir), "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug output: {debug_dir}")
    
    results = []
    missing = 0
    
    # #region agent log
    open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"F","location":"evaluate_model.py:216","message":"entering loop","data":{"gen_files_count":len(gen_files)},"timestamp":__import__('time').time()})+'\n')
    # #endregion
    
    for gen_file in gen_files:
        file_id = get_file_id(gen_file)
        
        # #region agent log
        open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"F","location":"evaluate_model.py:222","message":"loop iteration","data":{"gen_file":gen_file,"file_id":file_id,"in_fen_map":file_id in fen_map},"timestamp":__import__('time').time()})+'\n')
        # #endregion
        
        if file_id not in fen_map:
            print(f"Warning: No FEN found for {gen_file} (ID: {file_id})")
            missing += 1
            continue
        
        fen = fen_map[file_id]
        image_path = os.path.join(generated_dir, gen_file)
        
        # #region agent log
        open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"G","location":"evaluate_model.py:235","message":"before evaluate_single_image","data":{"image_path":image_path,"fen":fen[:30],"image_exists":os.path.exists(image_path)},"timestamp":__import__('time').time()})+'\n')
        # #endregion
        
        try:
            metrics = evaluate_single_image(
                image_path=image_path,
                fen=fen,
                sam_extractor=sam_extractor,
                threshold=threshold,
                debug_dir=debug_dir,
                file_id=file_id
            )
            
            # #region agent log
            open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"G","location":"evaluate_model.py:248","message":"after evaluate_single_image SUCCESS","data":{"cell_acc":metrics['cell_accuracy']},"timestamp":__import__('time').time()})+'\n')
            # #endregion
            
            results.append(metrics)
            print(f"  {gen_file}: cell_acc={metrics['cell_accuracy']:.2%}")
            
        except Exception as e:
            # #region agent log
            import traceback
            open('/home/avinoamd/roni/.cursor/debug.log','a').write(json.dumps({"hypothesisId":"H","location":"evaluate_model.py:256","message":"EXCEPTION in evaluate_single_image","data":{"error":str(e),"traceback":traceback.format_exc()},"timestamp":__import__('time').time()})+'\n')
            # #endregion
            print(f"Error processing {gen_file}: {e}")
            continue
    
    if results:
        avg_overall_acc = np.mean([r['overall_accuracy'] for r in results])
        avg_cell_acc = np.mean([r['cell_accuracy'] for r in results])
        avg_white_acc = np.mean([r['white_accuracy'] for r in results])
        avg_black_acc = np.mean([r['black_accuracy'] for r in results])
    else:
        avg_overall_acc = avg_cell_acc = avg_white_acc = avg_black_acc = 0.0
    
    summary = {
        'num_evaluated': len(results),
        'num_missing': missing,
        'avg_overall_accuracy': avg_overall_acc,
        'avg_cell_accuracy': avg_cell_acc,
        'avg_white_accuracy': avg_white_acc,
        'avg_black_accuracy': avg_black_acc,
        'error_rate': 1 - avg_cell_acc
    }
    
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Images Evaluated: {summary['num_evaluated']}")
    print(f"Average Cell Accuracy: {summary['avg_cell_accuracy']:.2%}")
    print(f"Average White Accuracy: {summary['avg_white_accuracy']:.2%}")
    print(f"Average Black Accuracy: {summary['avg_black_accuracy']:.2%}")
    print(f"Error Rate: {summary['error_rate']:.2%}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate chess board generation model using SAM segmentation"
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        default="/home/avinoamd/roni/evaluation/images_to_eval/model_output",
        help="Path to folder containing generated images"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/avinoamd/roni/BBDM/data_10.01_no_hands/val/val_data.csv",
        help="Path to CSV containing FEN ground truth"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/avinoamd/roni/BBDM/SAM/sam3.pt",
        help="Path to SAM checkpoint"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Coverage threshold for mask-to-grid conversion"
    )
    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="Disable debug output saving"
    )
    
    args = parser.parse_args()
    
    evaluate_folder(
        generated_dir=args.generated_dir,
        csv_path=args.csv_path,
        model_path=args.model_path,
        threshold=args.threshold,
        save_debug=not args.no_debug
    )


if __name__ == "__main__":
    main()
