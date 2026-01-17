import os
import subprocess
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scripts.crop_board import process_single_image
import sys
import yaml

# Add BBDM to Python path
sys.path.insert(0, "BBDM")
from BBDM.utils import dict2namespace
from BBDM.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel

# ==========================================
# Paths and Model Setup (According to the computer environment)
# ==========================================
BLENDER_EXEC = "/Applications/Blender.app/Contents/MacOS/Blender" # Path to your Blender executable
BLEND_FILE = "blender/chess-set.blend" # Blender file with the chess set
SCRIPT_FILE = "blender/generate_synthtic_from_fen.py" # The script we wrote earlier

# Device setup (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Model Loading
# ==========================================
def load_bbdm_model():
    """Load the trained LBBDM model for inference."""
    
    # Paths to your trained model
    config_path = "results/all_data_f4/LBBDM-f4/checkpoint/config.yaml"
    model_checkpoint = "results/all_data_f4/LBBDM-f4/checkpoint/last_model.pth"
    # Alternative: use best model instead
    # model_checkpoint = "results/all_data_f4/LBBDM-f4/checkpoint/top_model_epoch_84.pth"
    
    print(f"🔧 Loading LBBDM model from {model_checkpoint}...")
    
    # Load config (UnsafeLoader returns Namespace directly, no conversion needed)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    # Update paths to be relative to current directory
    config.model.VQGAN.params.ckpt_path = "results/VQGAN/last.ckpt"
    
    # Initialize model
    model = LatentBrownianBridgeModel(config.model).to(device)
    
    # Load trained weights
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # Set to evaluation mode
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, config

# Load model once at startup
bbdm_model, bbdm_config = load_bbdm_model()

# ==========================================
# Loading Your Model (BBDM)
# ==========================================

def generate_chessboard_image(fen: str, viewpoint: str) -> None:
    """
    Generate synthetic and realistic chessboard images from a given FEN.
    According to Project 3 specifications[cite: 436, 438].
    """
    
    # 1. Check viewpoint validity
    if viewpoint not in ['white', 'black']:
        raise ValueError("viewpoint must be 'white' or 'black'")
        
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True) # [cite: 440, 446]

    # Final file paths [cite: 441, 443, 444]
    path_synthetic = os.path.join(results_dir, "synthetic.png")
    path_realistic = os.path.join(results_dir, "realistic.png")
    path_sbs = os.path.join(results_dir, "side_by_side.png")

    # ======================================================
    # Step 1: Generate Synthetic Image (Using Blender) [cite: 441]
    # ======================================================
    # Blender always generates from white's perspective
    # We'll rotate 180° later if viewpoint is black

    print(f"🎨 Generating Synthetic Image for Viewpoint: {viewpoint}...")
    
    cmd = [
        BLENDER_EXEC,
        "-b", BLEND_FILE,
        "-P", SCRIPT_FILE,
        "--",
        "--fen", fen,
        "--output_dir", results_dir,
        "--output_name", "synthetic",  # Blender will add .png automatically
    ]
    
    try:
        # Run the command and suppress output (unless there's an error)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Blender: {e}")
        return

    if not os.path.exists(path_synthetic):
        print("❌ Error: Synthetic image was not created by Blender.")
        return

    # ======================================================
    # Step 1.5: Crop the Synthetic Image to Board Area Only
    # ======================================================
    # print("✂️  Cropping synthetic image to chessboard area...")
    
    # # Use the existing crop function (overwrites the original by default)
    # process_single_image(path_synthetic, output_dir=None, preview_mode=False)

    # ======================================================
    # Step 2: Generate Realistic Image (Using Your Model) [cite: 443]
    # ======================================================
    print("🤖 Generating Realistic Image using Neural Network...")
    
    # Load the synthetic image
    syn_image = Image.open(path_synthetic).convert('RGB')

    # Pre-processing 
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # or the size your model expects
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(syn_image).unsqueeze(0).to(device)

    # Run the model (Inference)
    with torch.no_grad():
        # === Use your trained BBDM model ===
        # The model expects input in range [-1, 1] which we already have
        # bbdm_model.sample() takes the synthetic image and returns realistic version
        fake_image_tensor = bbdm_model.sample(
            input_tensor, 
            clip_denoised=bbdm_config.testing.clip_denoised,
            sample_mid_step=False
        )
        # =================================================================

    # Post-processing
    fake_image = fake_image_tensor.squeeze().cpu().detach().numpy()
    fake_image = (fake_image + 1) / 2.0 * 255.0 # Denormalize
    fake_image = fake_image.transpose(1, 2, 0).astype(np.uint8)

    # Save the realistic image (ensure it matches the synthetic size if needed)
    # Here we save it in the original size of the synthetic for safety
    orig_w, orig_h = syn_image.size
    real_img_pil = Image.fromarray(fake_image).resize((orig_w, orig_h))
    real_img_pil.save(path_realistic)

    # ======================================================
    # Step 2.5: Rotate Realistic Image if Viewpoint is Black
    # ======================================================
    if viewpoint == 'black':
        print("🔄 Rotating realistic image 180 degrees (black viewpoint)...")
        
        # Load the realistic image
        img_to_rotate = cv2.imread(path_realistic)
        
        # Rotate 180 degrees
        rotated_img = cv2.rotate(img_to_rotate, cv2.ROTATE_180)
        
        # Save back to path_realistic (replacing the unrotated version)
        cv2.imwrite(path_realistic, rotated_img)
        print("✅ Realistic image rotated and saved.")

    # ======================================================
    # Step 3: Create Comparison Image (Side-by-Side) [cite: 444]
    # ======================================================
    print("🖼️ Creating Side-by-Side comparison...")
    
    img_syn = cv2.imread(path_synthetic)
    img_real = cv2.imread(path_realistic)

    # Ensure same sizes
    if img_syn.shape != img_real.shape:
        img_real = cv2.resize(img_real, (img_syn.shape[1], img_syn.shape[0]))

    # Horizontal stacking (Left: Synthetic, Right: Realistic)
    sbs_image = np.hstack((img_syn, img_real))

    # Save
    cv2.imwrite(path_sbs, sbs_image)
    
    print("✅ All images saved successfully in ./results/")

# ==========================================
# Self-Check (To Ensure It's Working)
# ==========================================
if __name__ == "__main__":
    # Example for Sicilian Opening from Black's Perspective
    test_fen = "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1"
    test_view = "white" 
    
    try:
        generate_chessboard_image(test_fen, test_view)
    except Exception as e:
        print(f"FAILED: {e}")