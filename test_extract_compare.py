import os
import cv2
import torch
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from evaluation.sam_mask_extractor import SAMMaskExtractor

# Config
IMAGE_PATH = "/home/avinoamd/roni/evaluation/images_to_eval/gt/game4_frame_037264.png"
OUTPUT_DIR = "/home/avinoamd/roni/evaluation/comparison_debug"
MODEL_PATH = "/home/avinoamd/roni/BBDM/SAM/sam3.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Processing image: {IMAGE_PATH}")

# ==============================================================================
# Method 1: Logic from BBDM/SAM/extract_chess_pieces.py
# ==============================================================================
print("\n--- Method 1: extract_chess_pieces.py logic ---")
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=MODEL_PATH,
    half=True,
    save=False,
)
predictor_1 = SAM3SemanticPredictor(overrides=overrides)
PROMPTS_1 = ["chess piece", "hand"]

predictor_1.set_image(IMAGE_PATH)
results_1 = predictor_1(text=PROMPTS_1)
result_1 = results_1[0]

# Logic from script: Class 0 is 'chess piece'
if result_1.masks is not None:
    all_masks = result_1.masks.data
    all_classes = result_1.boxes.cls.cpu().numpy().astype(int)
    
    chess_indices = np.where(all_classes == 0)[0]
    
    if len(chess_indices) > 0:
        chess_masks = all_masks[chess_indices]
        combined_mask = torch.any(chess_masks, dim=0).int() * 255
        mask_image_1 = combined_mask.cpu().numpy().astype(np.uint8)
        
        cv2.imwrite(os.path.join(OUTPUT_DIR, "method1_chess_piece_mask.png"), mask_image_1)
        print("Saved method1_chess_piece_mask.png")
    else:
        print("Method 1 found no chess pieces (Class 0)")
else:
    print("Method 1 returned no masks")


# ==============================================================================
# Method 2: Logic from evaluation/sam_mask_extractor.py
# ==============================================================================
print("\n--- Method 2: sam_mask_extractor.py logic ---")
extractor_2 = SAMMaskExtractor(model_path=MODEL_PATH, conf=0.25)
# Uses defaults: ["white chess piece", "black chess piece"]
mask_white, mask_black = extractor_2.extract_masks(IMAGE_PATH)

cv2.imwrite(os.path.join(OUTPUT_DIR, "method2_white_mask.png"), mask_white.astype(np.uint8) * 255)
cv2.imwrite(os.path.join(OUTPUT_DIR, "method2_black_mask.png"), mask_black.astype(np.uint8) * 255)

# Combined mask for easier comparison with Method 1
combined_2 = np.logical_or(mask_white, mask_black).astype(np.uint8) * 255
cv2.imwrite(os.path.join(OUTPUT_DIR, "method2_combined_mask.png"), combined_2)

print("Saved method2_white_mask.png")
print("Saved method2_black_mask.png")
print("Saved method2_combined_mask.png")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
