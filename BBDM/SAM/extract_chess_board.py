import os
import cv2
import torch
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from pathlib import Path

# ============ CONFIGURATION ============
IMAGE_PATH = "/home/avinoamd/roni/BBDM/data_10.01_no_hands/test/A/game2_frame_000896.png"  # Input image path
OUTPUT_PATH = "/home/avinoamd/roni/BBDM/SAM/game2_frame_000896_cropped.png"  # Output path (None = auto-generate as <input>_cropped.png)
PADDING = 0  # Pixels of padding around detected chessboard
# =======================================


def extract_chessboard(image_path: str, output_path: str = None, padding: int = 10) -> np.ndarray:
    """
    Extract and crop the chessboard from an image using SAM3.
    
    Detects all chess squares, creates a union of their masks, and crops
    to the bounding box of that union (excluding rim and background).
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save the cropped image
        padding: Pixels of padding around the detected board (default: 10)
    
    Returns:
        Cropped image as numpy array, or None if no chessboard found
    """
    # Initialize the predictor
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=True,
        save=False,
    )
    
    predictor = SAM3SemanticPredictor(overrides=overrides)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    
    # Set image for predictor
    predictor.set_image(image_path)
    
    # Run inference with chess squares prompt
    prompts = ["chess squares"]
    results = predictor(text=prompts)
    
    if not results or len(results) == 0:
        print("No results returned from predictor.")
        return None
    
    result = results[0]
    
    # Check if we have masks
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        print("No chess squares detected in the image.")
        return None
    
    # Get all masks and combine them into a union
    all_masks = result.masks.data  # Shape: (N, H, W)
    print(f"Found {len(all_masks)} chess square masks")
    
    # Create union of all masks
    combined_mask = torch.any(all_masks, dim=0).cpu().numpy().astype(np.uint8)
    
    # Find bounding box of the combined mask
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    
    if not rows.any() or not cols.any():
        print("Combined mask is empty.")
        return None
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    print(f"Chess squares bounding box: ({x1}, {y1}) to ({x2}, {y2})")
    
    # Apply padding while staying within image bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Crop the image
    cropped = image[y1:y2, x1:x2]
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        cv2.imwrite(output_path, cropped)
        print(f"Saved cropped chessboard to: {output_path}")
    
    return cropped


if __name__ == "__main__":
    # Generate default output path if not provided
    output_path = OUTPUT_PATH
    if output_path is None:
        input_path = Path(IMAGE_PATH)
        output_path = str(input_path.parent / f"{input_path.stem}_cropped.png")
    
    # Extract chessboard
    cropped = extract_chessboard(IMAGE_PATH, output_path, PADDING)
    
    if cropped is not None:
        print(f"Successfully extracted chessboard. Shape: {cropped.shape}")
    else:
        print("Failed to extract chessboard from image.")
        exit(1)
