import os
import glob
import cv2
import torch
import shutil
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from pathlib import Path
import sys

# 1. Setup paths
BASE_IMAGE_DIR = "/home/avinoamd/roni/BBDM/training_model_new"

# 2. Initialize the Predictor
PROMPTS = ["chess piece", "hand"]

# Resolve model path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "sam3.pt")

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=MODEL_PATH,
    half=True,
    save=False,
)

print(f"Loading model from {MODEL_PATH}...")
try:
    predictor = SAM3SemanticPredictor(overrides=overrides)
except Exception as e:
    print(f"Failed to initialize predictor: {e}")
    sys.exit(1)

# 3. Get list of all images recursively
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_paths = []

for ext in image_extensions:
    image_paths.extend(list(Path(BASE_IMAGE_DIR).rglob(ext)))

print(f"Found {len(image_paths)} images in {BASE_IMAGE_DIR}. Starting processing...")

for img_path_obj in image_paths:
    img_path = str(img_path_obj)
    filename = os.path.basename(img_path)
    
    try:
        path_parts = list(Path(img_path).relative_to(BASE_IMAGE_DIR).parts)
    except ValueError:
        print(f"Path error for {img_path}")
        continue

    # Expected structure: split / subfolder / filename  (e.g., train/A/img.jpg)
    if len(path_parts) >= 3: 
        split_name = path_parts[0] # test, train, val
        subfolder_name = path_parts[1] # A, B
        
        # Configure behavior based on subfolder
        if subfolder_name == 'B':
            # Real Data Logic
            target_mask_folder = "masks"
            
            masks_dir = os.path.join(BASE_IMAGE_DIR, split_name, target_mask_folder)
            os.makedirs(masks_dir, exist_ok=True)
            
        elif subfolder_name == 'A':
            # Synthetic Data Logic
            target_mask_folder = "mask_A"
            
            masks_dir = os.path.join(BASE_IMAGE_DIR, split_name, target_mask_folder)
            os.makedirs(masks_dir, exist_ok=True)
            
        else:
            # Skip folders that are not A or B
            continue
        
    else:
        # Unexpected structure
        continue
    
    print(f"Processing: {os.path.join(split_name, subfolder_name, filename)}")

    # --- Load Image ---
    try:
        predictor.set_image(img_path)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        continue
    
    # --- Run Inference ---
    try:
        results = predictor(text=PROMPTS)
    except Exception as e:
        print(f"Inference error on {img_path}: {e}")
        continue

    if not results:
        print("No results returned.")
        continue
        
    result = results[0]

    # --- Check for "Hand" (Class ID 1) ---
    hand_detected = False
    
    if result.boxes is not None and result.boxes.cls is not None:
        detected_classes = result.boxes.cls.cpu().numpy().astype(int)
        
        if 1 in detected_classes:
            hand_detected = True
            print(f"⚠️  Hand detected in {filename}!")

    # --- Extract & Save "Chess Piece" Masks (Class ID 0) ---
    # Extract masks regardless of hand detection
    if result.masks is not None:
        all_masks = result.masks.data
        all_classes = result.boxes.cls.cpu().numpy().astype(int)
        
        chess_indices = np.where(all_classes == 0)[0]
        
        if len(chess_indices) > 0:
            chess_masks = all_masks[chess_indices]
            # Combine all chess piece masks into one
            combined_mask = torch.any(chess_masks, dim=0).int() * 255
            mask_image = combined_mask.cpu().numpy().astype(np.uint8)
            
            # Save mask
            mask_filename = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(masks_dir, mask_filename)
            cv2.imwrite(save_path, mask_image)
            print(f"Saved merged chess mask to {target_mask_folder}.")
        else:
            print("No chess pieces found.")
    else:
        print("No detections/masks available.")

print("\nProcessing complete.")
