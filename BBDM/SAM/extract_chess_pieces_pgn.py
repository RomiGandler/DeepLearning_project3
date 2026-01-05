import os
import glob
import cv2
import torch
import shutil
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from pathlib import Path

# 1. Setup paths
BASE_IMAGE_DIR = "/home/avinoamd/roni/BBDM/pgn_data/game11-20260104T231534Z-3-001/game11/images"
SAM_MODEL_PATH = "/home/avinoamd/roni/BBDM/SAM/sam3.pt"

# 2. Initialize the Predictor
PROMPTS = ["chess piece", "hand"]

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=SAM_MODEL_PATH,
    half=True,
    save=False,
)

# Initialize predictor only once
try:
    predictor = SAM3SemanticPredictor(overrides=overrides)
except Exception as e:
    print(f"Error initializing SAM3 predictor: {e}")
    exit(1)

# 3. Get list of all images recursively
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_paths = []

print("Scanning for images...")
for ext in image_extensions:
    image_paths.extend(list(Path(BASE_IMAGE_DIR).rglob(ext)))

print(f"Found {len(image_paths)} images. Starting processing...")

for img_path_obj in image_paths:
    img_path = str(img_path_obj)
    filename = os.path.basename(img_path)
    
    # Check if 'images' is in the path to determine structure
    path_parts = list(Path(img_path).parts)
    
    try:
        # Find the 'images' directory index to determine where to put outputs
        # We search from the right to find the closest 'images' folder
        images_idx = len(path_parts) - 1 - path_parts[::-1].index('images')
        
        # Base output directory is the parent of 'images'
        # e.g. .../game13/images/img.jpg -> .../game13/
        output_base = os.path.join(*path_parts[:images_idx])
        # On Windows this join might lose the drive letter if split/joined incorrectly, 
        # but here on Linux dealing with absolute paths, it should be fine if we keep the root.
        # Actually, Path.parts includes root as first element '/'. 
        # os.path.join(*path_parts) should work for absolute paths on Linux.
        output_base = Path(*path_parts[:images_idx])
        
    except ValueError:
        # 'images' not found in path
        print(f"Skipping {img_path}: 'images' directory not found in path structure.")
        continue

    current_with_hand_dir = output_base / "with_hand"
    current_no_hand_dir = output_base / "no_hand"
    current_masks_dir = output_base / "masks"
    
    os.makedirs(current_with_hand_dir, exist_ok=True)
    os.makedirs(current_no_hand_dir, exist_ok=True)
    os.makedirs(current_masks_dir, exist_ok=True)
    
    print(f"Processing: {filename}")

    # --- Load Image ---
    try:
        predictor.set_image(img_path)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        continue
    
    # --- Run Inference ---
    results = predictor(text=PROMPTS)
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
            shutil.copy(img_path, current_with_hand_dir / filename)
        else:
            print(f"No hand detected in {filename}.")
            shutil.copy(img_path, current_no_hand_dir / filename)
    else:
        print(f"No detections at all in {filename}. Treating as no hand.")
        shutil.copy(img_path, current_no_hand_dir / filename)
            
    # --- Extract & Save "Chess Piece" Masks (Class ID 0) ---
    if not hand_detected and result.masks is not None:
        all_masks = result.masks.data
        all_classes = result.boxes.cls.cpu().numpy().astype(int)
        
        chess_indices = np.where(all_classes == 0)[0]
        
        if len(chess_indices) > 0:
            chess_masks = all_masks[chess_indices]
            # Combine all chess piece masks
            combined_mask = torch.any(chess_masks, dim=0).int() * 255
            mask_image = combined_mask.cpu().numpy().astype(np.uint8)
            
            # Save mask
            mask_filename = os.path.splitext(filename)[0] + ".png"
            save_path = current_masks_dir / mask_filename
            cv2.imwrite(str(save_path), mask_image)
            print(f"Saved merged chess mask.")
        else:
            print("No chess pieces found.")
    elif hand_detected:
         print("Hand detected - skipping mask generation.")
    else:
        print("No detections/masks available.")

print("\nProcessing complete.")

