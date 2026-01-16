import os
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

# Constants
BOARD_SIZE = 8
PIECE_WHITE = 1
PIECE_BLACK = 2
EMPTY = 0

def fen_to_grid(fen):
    """
    Parses a FEN string into an 8x8 grid.
    
    Grid values:
    0: Empty
    1: White Piece
    2: Black Piece
    """
    grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    
    # FEN placement is the first part of the string
    placement = fen.split(' ')[0]
    
    rows = placement.split('/')
    
    for r_idx, row in enumerate(rows):
        c_idx = 0
        for char in row:
            if char.isdigit():
                # Empty squares
                num_empty = int(char)
                c_idx += num_empty
            else:
                # Piece
                if char.isupper():
                    grid[r_idx, c_idx] = PIECE_WHITE
                else:
                    grid[r_idx, c_idx] = PIECE_BLACK
                c_idx += 1
                
    return grid

def test_fen_parser():
    # Starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    grid = fen_to_grid(start_fen)
    print("Testing FEN parser with starting position:")
    print(grid)
    
    assert grid[0, 0] == PIECE_BLACK # r
    assert grid[7, 0] == PIECE_WHITE # R
    assert grid[3, 3] == EMPTY
    print("Test passed!")

class ChessPredictor:
    def __init__(self, model_path="BBDM/SAM/sam3.pt", device='cuda'):
        self.device = device
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model=model_path,
            half=True,
            save=False,
            device=device
        )
        try:
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
        except Exception as e:
            print(f"Error initializing SAM predictor: {e}")
            self.predictor = None
            
        self.prompts = ["white chess piece", "black chess piece"]

    def predict_grid(self, image_path, threshold=0.20, debug_dir=None, file_id=None):
        """
        Runs SAM inference and maps masks to an 8x8 grid.
        Returns an 8x8 grid (0=Empty, 1=White, 2=Black).
        """
        if self.predictor is None:
            return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        try:
            self.predictor.set_image(image_path)
            results = self.predictor(text=self.prompts)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        if not results:
            return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        result = results[0]
        
        # Initialize masks
        if result.masks is not None:
            h, w = result.masks.data.shape[1:]
        else:
            h, w = result.orig_shape

        mask_white = np.zeros((h, w), dtype=bool)
        mask_black = np.zeros((h, w), dtype=bool)

        if result.masks is not None:
            # Class ID 0 corresponds to first prompt "white chess piece"
            # Class ID 1 corresponds to second prompt "black chess piece"
            
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy().astype(bool)

            for i, cls_id in enumerate(classes):
                if cls_id == 0:
                    mask_white = np.logical_or(mask_white, masks[i])
                elif cls_id == 1:
                    mask_black = np.logical_or(mask_black, masks[i])
        
        # Save Debug Masks
        if debug_dir and file_id:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{file_id}_white_mask.png"), mask_white.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(debug_dir, f"{file_id}_black_mask.png"), mask_black.astype(np.uint8) * 255)

        # Map masks to 8x8 grid
        predicted_grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        
        # h, w are already set above
        cell_h = h / BOARD_SIZE
        cell_w = w / BOARD_SIZE
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                # Define cell coordinates
                y_start = int(r * cell_h)
                y_end = int((r + 1) * cell_h)
                x_start = int(c * cell_w)
                x_end = int((c + 1) * cell_w)
                
                # Extract cell crops from masks
                white_cell = mask_white[y_start:y_end, x_start:x_end]
                black_cell = mask_black[y_start:y_end, x_start:x_end]
                
                cell_area = (y_end - y_start) * (x_end - x_start)
                if cell_area == 0:
                    continue
                    
                white_coverage = np.sum(white_cell) / cell_area
                black_coverage = np.sum(black_cell) / cell_area
                
                # Determine occupancy
                # Prioritize higher coverage if both exceed threshold
                if white_coverage > threshold and white_coverage >= black_coverage:
                    predicted_grid[r, c] = PIECE_WHITE
                elif black_coverage > threshold and black_coverage > white_coverage:
                    predicted_grid[r, c] = PIECE_BLACK
                else:
                    predicted_grid[r, c] = EMPTY
        
        if debug_dir and file_id:
             np.savetxt(os.path.join(debug_dir, f"{file_id}_pred_grid.txt"), predicted_grid, fmt='%d')

        return predicted_grid

import re

def evaluate_folders(gt_dir, generated_dir, model_path, csv_path=None):
    print(f"Starting evaluation from folders")
    print(f"GT dir: {gt_dir}")
    print(f"Generated dir: {generated_dir}")
    if csv_path:
        print(f"Using CSV for GT FENs: {csv_path}")
    
    predictor = ChessPredictor(model_path=model_path)
    if predictor.predictor is None:
        print("Failed to initialize predictor. Aborting.")
        return

    # List files
    if not os.path.exists(generated_dir):
         print(f"Generated directory not found: {generated_dir}")
         return
    if not os.path.exists(gt_dir):
         print(f"GT directory not found: {gt_dir}")
         return

    gen_files = [f for f in os.listdir(generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Map IDs to GT filenames for easier lookup
    # Heuristic: extract the last sequence of digits
    def get_id(fname):
        nums = re.findall(r'\d+', fname)
        if nums:
            return nums[-1]
        return fname

    gt_map = {get_id(f): f for f in gt_files}
    
    # Load CSV if provided
    fen_map = {}
    if csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Find FEN column
            fen_col = None
            filename_col = None
            
            # Clean columns
            df.columns = [c.strip() for c in df.columns]
            
            if 'FEN' in df.columns:
                fen_col = 'FEN'
            else:
                fen_col = df.columns[0]
                
            for col in df.columns:
                if df[col].astype(str).str.contains('.png').any():
                    filename_col = col
                    break
            
            if fen_col and filename_col:
                for _, row in df.iterrows():
                    fname = str(row[filename_col])
                    fid = get_id(fname)
                    fen_map[fid] = row[fen_col]
            else:
                print("Could not identify FEN/Filename columns in CSV. Falling back to image-based GT.")
        except Exception as e:
             print(f"Error reading CSV: {e}")

    # Create debug directory
    debug_root = os.path.join(os.path.dirname(generated_dir), "debug_output")
    os.makedirs(debug_root, exist_ok=True)
    
    total_cells = 0
    correct_cells = 0
    total_images_processed = 0
    missing_images = 0
    image_accuracies = []

    for gen_file in gen_files:
        gen_id = get_id(gen_file)
        
        if gen_id not in gt_map:
            print(f"Warning: No matching GT image found for {gen_file} (ID: {gen_id})")
            missing_images += 1
            continue
            
        gt_file = gt_map[gen_id]
        gt_path = os.path.join(gt_dir, gt_file)
        gen_path = os.path.join(generated_dir, gen_file)
        
        print(f"Comparing {gt_file} (GT) vs {gen_file} (Gen)")

        # Determine GT Grid
        gt_grid = None
        if gen_id in fen_map:
            # Use FEN from CSV
            try:
                gt_grid = fen_to_grid(fen_map[gen_id])
                print("  Using GT from CSV FEN")
            except Exception as e:
                print(f"  Error parsing FEN for {gen_id}: {e}")
        
        if gt_grid is None:
            # Fallback: Use predictor on GT image (NOT IDEAL for strict eval, but useful if no CSV)
            print("  Warning: Using predictor on GT image as ground truth (No CSV/FEN found)")
            gt_grid = predictor.predict_grid(gt_path)

        # Save GT Grid
        np.savetxt(os.path.join(debug_root, f"{gen_id}_gt_grid.txt"), gt_grid, fmt='%d')

        # Predict Gen with Debug Saving
        pred_grid = predictor.predict_grid(gen_path, debug_dir=debug_root, file_id=gen_id)
        
        matches = (gt_grid == pred_grid)
        num_matches = np.sum(matches)
        
        image_acc = num_matches / (BOARD_SIZE * BOARD_SIZE)
        image_accuracies.append(image_acc)
        
        correct_cells += num_matches
        total_cells += (BOARD_SIZE * BOARD_SIZE)
        total_images_processed += 1
        
        print(f"  Accuracy: {image_acc:.2%}")

    print("\n" + "="*30)
    print("Folder Evaluation Complete")
    print(f"Images Processed: {total_images_processed}")
    print(f"Missing Images: {missing_images}")
    print(f"Debug output saved to: {debug_root}")
    
    if total_cells > 0:
        overall_accuracy = correct_cells / total_cells
        avg_image_accuracy = np.mean(image_accuracies) if image_accuracies else 0
        
        print(f"Overall Cell Accuracy: {overall_accuracy:.2%}")
        print(f"Average Image Accuracy: {avg_image_accuracy:.2%}")
        print(f"Overall Error Rate: {1 - overall_accuracy:.2%}")
    else:
        print("No images evaluated.")

def evaluate(generated_dir, data_root, split, model_path):
    print(f"Starting evaluation for split: {split}")
    print(f"Data root: {data_root}")
    print(f"Generated dir: {generated_dir}")
    
    # Load CSV
    csv_path = os.path.join(data_root, split, f"{split}_data.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        # Assuming CSV has no header based on previous file read, or we check content
        # The file read showed: FEN,FRAME,GAME,,
        # So it has a header.
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    predictor = ChessPredictor(model_path=model_path)
    if predictor.predictor is None:
        print("Failed to initialize predictor. Aborting.")
        return

    total_cells = 0
    correct_cells = 0
    
    # Breakdown of errors
    total_images_processed = 0
    missing_images = 0
    
    # Store per-image accuracy
    image_accuracies = []

    # Check columns - adapt if necessary. 
    # Based on read_file: FEN, FRAME, GAME, (filename col 1), (filename col 2)
    # It seems columns might be index-based if header parsing fails or names are weird.
    # Let's clean up column names if needed.
    df.columns = [c.strip() for c in df.columns]
    
    # Identify FEN column and Filename column
    # Based on preview: 
    # Col 0: FEN
    # Col 3/4: filename
    
    fen_col = None
    filename_col = None
    
    if 'FEN' in df.columns:
        fen_col = 'FEN'
    else:
        # Fallback to index 0
        fen_col = df.columns[0]
        
    # Find a column that looks like a filename (ends with .png)
    for col in df.columns:
        if df[col].astype(str).str.contains('.png').any():
            filename_col = col
            break
            
    if not fen_col or not filename_col:
        print(f"Could not identify FEN or filename columns in {csv_path}")
        return

    print(f"Using FEN column: {fen_col}, Filename column: {filename_col}")

    for idx, row in df.iterrows():
        fen = row[fen_col]
        filename = row[filename_col]
        
        # Check if generated image exists
        # Generated images might be directly in generated_dir or in subfolders?
        # The user query implies "folder of generated images". Let's assume flat or matching name.
        
        gen_img_path = os.path.join(generated_dir, filename)
        if not os.path.exists(gen_img_path):
            # Try searching recursively if structure differs? 
            # Or just warn as requested.
            # "if you encounter a fen you dont see the image it reffer to in the folder give a warning and continue"
            print(f"Warning: Generated image not found for {filename}")
            missing_images += 1
            continue

        # Get GT Grid
        try:
            gt_grid = fen_to_grid(fen)
        except Exception as e:
            print(f"Error parsing FEN for {filename}: {e}")
            continue
            
        # Get Predicted Grid
        pred_grid = predictor.predict_grid(gen_img_path)
        
        # Compare
        # Element-wise comparison
        matches = (gt_grid == pred_grid)
        num_matches = np.sum(matches)
        
        image_acc = num_matches / (BOARD_SIZE * BOARD_SIZE)
        image_accuracies.append(image_acc)
        
        correct_cells += num_matches
        total_cells += (BOARD_SIZE * BOARD_SIZE)
        total_images_processed += 1
        
        if idx % 10 == 0:
            print(f"Processed {idx+1}/{len(df)}: {filename} - Acc: {image_acc:.2%}")

    print("\n" + "="*30)
    print("Evaluation Complete")
    print(f"Images Processed: {total_images_processed}")
    print(f"Missing Images: {missing_images}")
    
    if total_cells > 0:
        overall_accuracy = correct_cells / total_cells
        avg_image_accuracy = np.mean(image_accuracies) if image_accuracies else 0
        
        print(f"Overall Cell Accuracy: {overall_accuracy:.2%}")
        print(f"Average Image Accuracy: {avg_image_accuracy:.2%}")
        
        # Error Score as requested (reduce score for mistakes)
        # This is implicitly 1 - accuracy, or maybe sum of errors.
        # "return the error/accurcy score"
        print(f"Overall Error Rate: {1 - overall_accuracy:.2%}")
    else:
        print("No images evaluated.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Chess Model Accuracy")
    parser.add_argument("--generated_dir", required=True, help="Path to generated images folder")
    # Mode selection
    parser.add_argument("--mode", choices=["csv", "folder"], default="csv", help="Evaluation mode: 'csv' (default) uses data_root/split/csv, 'folder' uses gt_dir")
    
    # CSV mode args
    parser.add_argument("--data_root", default="/home/avinoamd/roni/BBDM/data_10.01_no_hands", help="Path to data root containing split folders (for csv mode)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate (for csv mode)")
    
    # Folder mode args
    parser.add_argument("--gt_dir", help="Path to GT images folder (for folder mode)")
    parser.add_argument("--csv_path", help="Path to CSV containing FEN strings (optional for folder mode, but recommended for accurate GT)")
    
    parser.add_argument("--model_path", default="/home/avinoamd/roni/BBDM/SAM/sam3.pt", help="Path to SAM checkpoint")
    
    args = parser.parse_args()
    
    if args.mode == "folder":
        if not args.gt_dir:
            # Fallback or error? User might provide just generated_dir if they structured it carefully, 
            # but let's assume they provide both or we use a default structure relative to generated_dir?
            # User example: evaluation/images_to_eval/
            # If generated_dir ends in 'outputs', maybe gt_dir is '../gt'?
            # Let's be strict for now or try to infer.
            if os.path.basename(args.generated_dir) == "outputs":
                parent = os.path.dirname(args.generated_dir)
                args.gt_dir = os.path.join(parent, "gt")
                print(f"Inferred GT dir: {args.gt_dir}")
            else:
                 print("Error: --gt_dir required for folder mode")
                 return
        evaluate_folders(args.gt_dir, args.generated_dir, args.model_path, args.csv_path)
    else:
        evaluate(args.generated_dir, args.data_root, args.split, args.model_path)

if __name__ == "__main__":
    main()
