import os
import subprocess
import cv2
import numpy as np
import sys
import yaml
import torch
from PIL import Image

# ==========================================
# Imports from src/ and scripts/
# ==========================================
# Ensure current directory is in path
sys.path.append(os.getcwd())

from src.bbdm.inference import BBDMPipeline
from src.evaluation.evaluate_model import evaluate_single
from src.evaluation.sam_grid_extractor import SAMGridExtractor
from src.evaluation.data_saver import DataSaver
from src.blender.crop_board import process_single_image
from etl.augmentations.mask_extraction import SAMMaskExtractor

# ==========================================
# Helper Functions
# ==========================================

def _save_grid_visual(grid: np.ndarray, save_path: str, cell_size: int = 50):
    """
    Save a visual representation of the predicted grid.
    Colors: Green=White piece, Red=Black piece, Yellow=Both (conflict), Black=Empty
    """
    board_size = grid.shape[1]
    h, w = board_size * cell_size, board_size * cell_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for r in range(board_size):
        for c in range(board_size):
            y1, y2 = r * cell_size, (r + 1) * cell_size
            x1, x2 = c * cell_size, (c + 1) * cell_size
            
            is_white = grid[0, r, c] > 0.5
            is_black = grid[1, r, c] > 0.5
            
            if is_white and is_black:
                color = (0, 255, 255)  # Yellow - conflict
            elif is_white:
                color = (0, 255, 0)    # Green - white piece
            elif is_black:
                color = (0, 0, 255)    # Red - black piece
            else:
                color = (0, 0, 0)      # Black - empty
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 1)  # Grid lines
    
    cv2.imwrite(save_path, img)

# ==========================================
# Configuration Loading
# ==========================================
CONFIG_FILE = "submission_config.yaml"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Configuration file {CONFIG_FILE} not found!")
    with open(CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

# Global cache
_PIPELINE = None
_EXTRACTOR = None
_MASK_EXTRACTOR = None
_CONFIG = None

def get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG

def get_pipeline():
    global _PIPELINE
    cfg = get_config()
    if _PIPELINE is None:
        print("🔧 Loading BBDM Pipeline...")
        # Resolve paths relative to current working directory
        _PIPELINE = BBDMPipeline(
            config=cfg['models'].get('bbdm_config'),
            bbdm_checkpoint=cfg['models']['bbdm_checkpoint'],
            vqgan_checkpoint=cfg['models'].get('vqgan_checkpoint'),  # Optional for new checkpoints
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _PIPELINE

def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        print("🔧 Loading SAM Grid Extractor...")
        _EXTRACTOR = SAMGridExtractor()
    return _EXTRACTOR

def get_mask_extractor():
    global _MASK_EXTRACTOR
    if _MASK_EXTRACTOR is None:
        print("🔧 Loading SAM Mask Extractor...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MASK_EXTRACTOR = SAMMaskExtractor(device=device)
    return _MASK_EXTRACTOR

# ==========================================
# Main Function
# ==========================================

def generate_chessboard_image(fen: str, viewpoint: str) -> None:
    """
    Generate synthetic and realistic chessboard images from a given FEN.
    Reads settings from submission_config.yaml but enforces output to ./results/
    """
    cfg = get_config()
    
    # 1. Validation
    if viewpoint not in ['white', 'black']:
        raise ValueError("viewpoint must be 'white' or 'black'")
        
    # STRICT REQUIREMENT: Output must be in ./results/
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    path_synthetic = os.path.join(results_dir, "synthetic.png")
    path_realistic = os.path.join(results_dir, "realistic.png")
    path_sbs = os.path.join(results_dir, "side_by_side.png")

    # ======================================================
    # Step 1: Generate Synthetic Image (Using Blender)
    # ======================================================
    print(f"🎨 Generating Synthetic Image for Viewpoint: {viewpoint}...")
    
    blender_exec = cfg['blender']['exec_path']
    blend_file = cfg['blender']['blend_file']
    script_file = cfg['blender']['script_file']

    cmd = [
        blender_exec,
        "-b", blend_file,
        "-P", script_file,
        "--",
        "--fen", fen,
        "--output_dir", results_dir,
        "--output_name", "synthetic",
    ]
    
    try:
        # Run Blender
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        
        print(f"❌ Error running Blender: {e}")
        print(e.stdout)
        print(e.stderr)
        return
    except FileNotFoundError:
        print(f"❌ Blender executable not found at: {blender_exec}")
        return

    if not os.path.exists(path_synthetic):
        print("❌ Error: Synthetic image was not created by Blender.")
        return

    # ======================================================
    # Step 1.5: Crop the Synthetic Image
    # ======================================================
    print("✂️  Cropping synthetic image to chessboard area...")
    try:
        # Overwrites the original by default as per original script behavior
        process_single_image(path_synthetic, output_dir=None, preview_mode=False)
    except Exception as e:
        print(f"⚠️  Cropping failed: {e}")

    # ======================================================
    # Step 2: Generate Realistic Image (Using src/ Pipeline)
    # ======================================================
    print("🤖 Generating Realistic Image...")
    
    try:
        pipeline = get_pipeline()
        
        # For mask-guided models, extract masks using SAM
        masks = None
        if pipeline.is_mask_guided:
            print("📍 Extracting piece masks using SAM...")
            extractor = get_mask_extractor()
            image_bgr = cv2.imread(path_synthetic)
            black_mask, white_mask = extractor.extract_masks(image_bgr)
            if black_mask is not None and white_mask is not None:
                masks = torch.stack([
                    torch.from_numpy(white_mask.astype(np.float32)),
                    torch.from_numpy(black_mask.astype(np.float32))
                ], dim=0).unsqueeze(0)
            else:
                print("⚠️  SAM couldn't detect pieces, proceeding without masks...")
        
        clip_denoised = getattr(getattr(pipeline.config, 'testing', None), 'clip_denoised', False)
        
        real_img_pil = pipeline.generate_from_path(
            path_synthetic,
            masks=masks,
            clip_denoised=clip_denoised
        )
        real_img_pil.save(path_realistic)
    except Exception as e:
        print(f"❌ Error generating realistic image: {e}")
        return

    # ======================================================
    # Optional: Evaluation
    # ======================================================
    # Check config or env var (DISABLE_EVALUATION takes priority)
    if os.environ.get("DISABLE_EVALUATION") == "1":
        should_eval = False
    else:
        should_eval = cfg.get('evaluation', {}).get('enabled', False) or \
                      os.environ.get("ENABLE_EVALUATION") == "1"

    if should_eval:
        print("📊 Running Evaluation on generated image...")
        try:
            extractor = get_extractor()
            # Reload from disk to ensure we evaluate exactly what was saved
            real_img_cv2 = cv2.imread(path_realistic)
            if real_img_cv2 is None:
                 raise ValueError("Could not read realistic image for evaluation")
            
            # Use DataSaver to save predicted grid and other debug images
            saver = DataSaver(results_dir)
            
            metrics = evaluate_single(
                image=real_img_cv2,
                fen=fen,
                file_id="generated_eval",
                extractor=extractor,
                saver=saver
            )
            
            # Also save predicted grid visualization directly to results folder
            pred_grid = metrics['pred_grid']
            path_pred_grid = os.path.join(results_dir, "predicted_grid.png")
            _save_grid_visual(pred_grid, path_pred_grid)
            
            print(f"   ✅ Cell Accuracy: {metrics['cell_accuracy']:.2%}")
            print(f"   💾 Predicted grid saved to: {path_pred_grid}")
            print(f"   📁 Detailed debug files in: {os.path.join(saver.debug_dir, 'generated_eval')}/")
        except Exception as e:
            print(f"   ⚠️ Evaluation failed: {e}")

    # ======================================================
    # Step 3: Rotate Images if Viewpoint is Black
    # ======================================================
    if viewpoint == 'black':
        print("🔄 Rotating images 180 degrees (black viewpoint)...")
        
        for p in [path_synthetic, path_realistic]:
            img = cv2.imread(p)
            if img is not None:
                img = cv2.rotate(img, cv2.ROTATE_180)
                cv2.imwrite(p, img)
            else:
                print(f"⚠️  Could not rotate {p}: Image not found or invalid.")

    # ======================================================
    # Step 4: Create Side-by-Side Comparison
    # ======================================================
    print("🖼️ Creating Side-by-Side comparison...")
    
    img_syn = cv2.imread(path_synthetic)
    img_real = cv2.imread(path_realistic)

    if img_syn is not None and img_real is not None:
        if img_syn.shape != img_real.shape:
            # Resize realistic to match synthetic height/width
            img_real = cv2.resize(img_real, (img_syn.shape[1], img_syn.shape[0]))
        
        sbs_image = np.hstack((img_syn, img_real))
        cv2.imwrite(path_sbs, sbs_image)
        print("✅ All images saved successfully in ./results/")
    else:
        print("❌ Error reading images for side-by-side.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate realistic chessboard images from FEN notation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from starting position (white view)
  python submission.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
  
  # Generate from black's perspective
  python submission.py --fen "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1" --viewpoint black
  
  # Disable evaluation
  python submission.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" --no-eval
        """
    )
    
    parser.add_argument(
        "--fen", "-f",
        type=str,
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="FEN string describing the chess position (default: starting position)"
    )
    parser.add_argument(
        "--viewpoint", "-v",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Viewpoint: 'white' or 'black' (default: white)"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Override evaluation setting if --no-eval flag is set
    if args.no_eval:
        os.environ["DISABLE_EVALUATION"] = "1"
    
    try:
        print(f"🎯 FEN: {args.fen}")
        print(f"👁️  Viewpoint: {args.viewpoint}")
        generate_chessboard_image(args.fen, args.viewpoint)
    except Exception as e:
        print(f"❌ FAILED: {e}")
        raise
