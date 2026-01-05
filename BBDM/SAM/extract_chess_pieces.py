import os
import glob
import cv2
import torch
import shutil
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from pathlib import Path

# 1. Setup paths
BASE_IMAGE_DIR = "/home/avinoamd/roni/BBDM/friefeld_data"  # Base input directory
# We will create the output structure INSIDE the base directory as requested
# "save the folders of masks, no hand, with hand inside /home/avinoamd/roni/BBDM/friefeld_data/test and .../train and .../val"

# 2. Initialize the Predictor
PROMPTS = ["chess piece", "hand"]

overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
    save=False,
)

predictor = SAM3SemanticPredictor(overrides=overrides)

# 3. Get list of all images recursively
# We want to walk through the directory structure
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_paths = []

for ext in image_extensions:
    image_paths.extend(list(Path(BASE_IMAGE_DIR).rglob(ext)))

print(f"Found {len(image_paths)} images. Starting batch processing...")

for img_path_obj in image_paths:
    img_path = str(img_path_obj)
    filename = os.path.basename(img_path)
    
    path_parts = list(Path(img_path).relative_to(BASE_IMAGE_DIR).parts)
    # path_parts will be ['test', 'A', 'img.jpg'] or ['train', 'B', 'img.jpg']
    
    if len(path_parts) >= 3: # Must have at least split/subfolder/filename
        split_name = path_parts[0] # test, train, val
        subfolder_name = path_parts[1] # A, B
        
        if subfolder_name != 'B':
            # print(f"Skipping {img_path} (not in B folder)")
            continue
        
        current_with_hand_dir = os.path.join(BASE_IMAGE_DIR, split_name, "with_hand")
        current_no_hand_dir = os.path.join(BASE_IMAGE_DIR, split_name, "no_hand")
        current_masks_dir = os.path.join(BASE_IMAGE_DIR, split_name, "masks")
        
    else:
        # Fallback if structure is shallower than expected
        print(f"Skipping file with unexpected path structure: {img_path}")
        continue
    
    os.makedirs(current_with_hand_dir, exist_ok=True)
    os.makedirs(current_no_hand_dir, exist_ok=True)
    os.makedirs(current_masks_dir, exist_ok=True)
    
    print(f"\nProcessing: {os.path.join(split_name, subfolder_name, filename)}")

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
            shutil.copy(img_path, os.path.join(current_with_hand_dir, filename))
        else:
            print(f"No hand detected in {filename}.")
            shutil.copy(img_path, os.path.join(current_no_hand_dir, filename))
    else:
        print(f"No detections at all in {filename}. Treating as no hand.")
        shutil.copy(img_path, os.path.join(current_no_hand_dir, filename))
            
    # --- Extract & Save "Chess Piece" Masks (Class ID 0) ---
    # If you want masks for ALL images, remove the 'not hand_detected' check.
    if not hand_detected and result.masks is not None:
        all_masks = result.masks.data
        all_classes = result.boxes.cls.cpu().numpy().astype(int)
        
        chess_indices = np.where(all_classes == 0)[0]
        
        if len(chess_indices) > 0:
            chess_masks = all_masks[chess_indices]
            combined_mask = torch.any(chess_masks, dim=0).int() * 255
            mask_image = combined_mask.cpu().numpy().astype(np.uint8)
            
            # Save mask with same filename (or verify if extension needs change)
            # Usually masks are png
            mask_filename = os.path.splitext(filename)[0] + ".png"
            save_path = os.path.join(current_masks_dir, mask_filename)
            cv2.imwrite(save_path, mask_image)
            print(f"Saved merged chess mask.")
        else:
            print("No chess pieces found.")
    elif hand_detected:
         print("Hand detected - skipping mask generation.")
    else:
        print("No detections/masks available.")

print("\nProcessing complete.")