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

from src.bbdm.inference import load_pipeline
from src.evaluation.evaluate_model import evaluate_single
from src.evaluation.sam_grid_extractor import SAMGridExtractor

# Import the cropping script as requested
# We assume 'scripts/crop_board.py' exists as per instructions
try:
    from scripts.crop_board import process_single_image
except ImportError:
    print("⚠️  Warning: Could not import scripts.crop_board. Cropping may fail if script is missing.")
    # Define a dummy function to avoid NameError if import fails but logic continues
    def process_single_image(*args, **kwargs):
        print("❌ Error: process_single_image called but module not loaded.")

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
        _PIPELINE = load_pipeline(
            config=cfg['models']['bbdm_config'],
            bbdm_checkpoint=cfg['models']['bbdm_checkpoint'],
            vqgan_checkpoint=cfg['models']['vqgan_checkpoint'],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _PIPELINE

def get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        print("🔧 Loading SAM Grid Extractor...")
        _EXTRACTOR = SAMGridExtractor()
    return _EXTRACTOR

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
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Blender: {e}")
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
        
        real_img_pil = pipeline.generate_from_path(
            path_synthetic, 
            clip_denoised=pipeline.config.testing.clip_denoised
        )
        real_img_pil.save(path_realistic)
    except Exception as e:
        print(f"❌ Error generating realistic image: {e}")
        return

    # ======================================================
    # Optional: Evaluation
    # ======================================================
    # Check config or env var
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
            
            metrics = evaluate_single(
                image=real_img_cv2,
                fen=fen,
                file_id="generated_eval",
                extractor=extractor
            )
            print(f"   ✅ Cell Accuracy: {metrics['cell_accuracy']:.2%}")
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
    # Example usage for testing
    test_fen = "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1"
    try:
        generate_chessboard_image(test_fen, "white")
    except Exception as e:
        print(f"FAILED: {e}")
