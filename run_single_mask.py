import os
import cv2
import torch
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

# Config
IMAGE_PATH = "/home/avinoamd/roni/BBDM/results/all_data_f4/LBBDM-f4/evaluation/gt/debug/game2_frame_001216/gt_original.png"
OUTPUT_PATH = "/home/avinoamd/roni/gt_with_mask_annotated.png"
MODEL_PATH = "/home/avinoamd/roni/BBDM/SAM/sam3.pt"

print(f"Processing image: {IMAGE_PATH}")

overrides = dict(
    conf=0.5,
    task="segment",
    mode="predict",
    model=MODEL_PATH,
    half=True,
    save=False,
)
predictor = SAM3SemanticPredictor(overrides=overrides)
PROMPTS = ["chess piece - pawn"]

predictor.set_image(IMAGE_PATH)
results = predictor(text=PROMPTS)
result = results[0]

# Show all detected classes
if result.masks is not None:
    all_masks = result.masks.data
    all_classes = result.boxes.cls.cpu().numpy().astype(int)
    all_confidences = result.boxes.conf.cpu().numpy()
    
    n_detections = len(all_masks)
    
    if n_detections > 0:
        # Load original image
        original = cv2.imread(IMAGE_PATH)
        output_img = original.copy()
        
        # Generate distinct colors for each detection
        colors = []
        for i in range(n_detections):
            hue = int(180 * i / n_detections)
            hsv = np.uint8([[[hue, 200, 220]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in bgr))
        
        # Overlay each mask with its own color
        for i, mask in enumerate(all_masks):
            mask_np = mask.cpu().numpy().astype(np.uint8)
            mask_overlay = np.zeros_like(original)
            mask_overlay[mask_np > 0] = colors[i]
            output_img = cv2.addWeighted(output_img, 1.0, mask_overlay, 0.15, 0)
        
        # Print detections grouped by class
        print(f"\nDetections ({n_detections} total):")
        for cls_id in sorted(set(all_classes)):
            cls_name = PROMPTS[cls_id] if cls_id < len(PROMPTS) else f"class_{cls_id}"
            print(f"\n  Class {cls_id}: \"{cls_name}\"")
            
            for i, (mask, conf, cls) in enumerate(zip(all_masks, all_confidences, all_classes)):
                if cls != cls_id:
                    continue
                mask_np = mask.cpu().numpy().astype(np.uint8)
                y_coords, x_coords = np.where(mask_np > 0)
                
                if len(x_coords) > 0:
                    centroid_x = int(np.mean(x_coords))
                    centroid_y = int(np.mean(y_coords))
                    
                    # Draw compact ID at centroid with matching color
                    cv2.putText(output_img, str(i), (centroid_x, centroid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, colors[i], 1)
                    
                    print(f"    [{i}] conf={conf:.3f}  pos=({centroid_x},{centroid_y})")
        
        cv2.imwrite(OUTPUT_PATH, output_img)
        print(f"\nSaved annotated image with {n_detections} detections to: {OUTPUT_PATH}")
    else:
        print("No detections found")
else:
    print("No masks returned")
