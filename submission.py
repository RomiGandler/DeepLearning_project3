import os
import subprocess
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# ==========================================
# Paths and Model Setup (According to the computer environment)
# ==========================================
BLENDER_EXEC = "/Applications/Blender.app/Contents/MacOS/Blender" # Path to your Blender executable
BLEND_FILE = "blender/chess-set.blend" # Blender file with the chess set
SCRIPT_FILE = "blender/generate_synthtic_from_fen.py" # The script we wrote earlier

# Device setup (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # We call the Blender script we created as a subprocess
    # Mapping: viewpoint='white' -> side='white', viewpoint='black' -> side='black'

    print(f"🎨 Generating Synthetic Image for Viewpoint: {viewpoint}...")
    
    cmd = [
        BLENDER_EXEC,
        "-b", BLEND_FILE,
        "-P", SCRIPT_FILE,
        "--",
        "--fen", fen,
        "--output_dir", results_dir,
        "--output_name", "synthetic",  # Blender will add .png automatically
        "--side", viewpoint # Here we pass the viewpoint [cite: 439, 445]
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
    # Step 2: Generate Realistic Image (Using Your Model) [cite: 443]
    # ======================================================
    print("🤖 Generating Realistic Image using Neural Network...")
    
    # Load the synthetic image
    syn_image = Image.open(path_synthetic).convert('RGB')

    # Pre-processing - Must match what you did during training!
    # Common example for CycleGAN:
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # or the size your model expects
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(syn_image).unsqueeze(0).to(device)

    # Run the model (Inference)
    with torch.no_grad():
        # === Critical Line: Here your model generates the image ===
        # fake_image_tensor = netG(input_tensor)

        # --- For now (until you connect the model): We'll use the synthetic image as a dummy ---
        fake_image_tensor = input_tensor 
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
    test_fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    test_view = "black" 
    
    try:
        generate_chessboard_image(test_fen, test_view)
    except Exception as e:
        print(f"FAILED: {e}")