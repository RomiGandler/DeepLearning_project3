#!/usr/bin/env python3
"""
Test script to visualize BBDM and VQGAN preprocessing side-by-side.

Verifies that both pipelines produce identical results after our fixes.
"""

from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def main():
    # === MODIFY THIS ===
    IMAGE_PATH = "/Users/avinoamd/deg/roni/DeepLearning_project3/chess_data/test/B/game12_frame_029896.png"  # Path to test image
    OUTPUT_DIR = None  # Output directory (default: same dir as image)
    SIZE = 256         # Target size
    # ===================

    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    pil_img = Image.open(image_path).convert("RGB")

    print(f"Input image: {image_path}")
    print(f"Original size: {pil_img.size[0]}x{pil_img.size[1]}")
    print(f"Target size: {SIZE}x{SIZE}")
    print()

    # === BBDM preprocessing (torchvision) ===
    bbdm_transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor()
    ])
    bbdm_tensor = bbdm_transform(pil_img)  # [C, H, W] in [0, 1]
    bbdm_array = (bbdm_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # === VQGAN preprocessing (PIL resize, same as BBDM now) ===
    vqgan_pil_resized = pil_img.resize((SIZE, SIZE), Image.BILINEAR)
    vqgan_array = np.array(vqgan_pil_resized)  # [H, W, C] in [0, 255]

    # === Compare ===
    diff = np.abs(bbdm_array.astype(np.int16) - vqgan_array.astype(np.int16))
    max_diff = diff.max()
    mean_diff = diff.mean()

    print("=== Comparison ===")
    print(f"BBDM output shape: {bbdm_array.shape}")
    print(f"VQGAN output shape: {vqgan_array.shape}")
    print(f"Max pixel difference: {max_diff}")
    print(f"Mean pixel difference: {mean_diff:.4f}")

    if max_diff <= 1:
        print("✓ Outputs are essentially identical (diff ≤ 1 due to rounding)")
    else:
        print("✗ WARNING: Outputs differ significantly!")
    print()

    # === Save visualizations ===
    stem = image_path.stem

    # Original (resized for comparison panel)
    original_resized = np.array(pil_img.resize((SIZE, SIZE), Image.BILINEAR))

    # Save individual outputs
    cv2.imwrite(str(output_dir / f"{stem}_bbdm.png"), cv2.cvtColor(bbdm_array, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / f"{stem}_vqgan.png"), cv2.cvtColor(vqgan_array, cv2.COLOR_RGB2BGR))

    # Save diff visualization (amplified for visibility)
    diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / f"{stem}_diff_amplified.png"), diff_vis)

    # Save side-by-side comparison
    comparison = np.hstack([bbdm_array, vqgan_array, diff_vis])
    cv2.imwrite(str(output_dir / f"{stem}_comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print("=== Saved ===")
    print(f"  {stem}_bbdm.png")
    print(f"  {stem}_vqgan.png")
    print(f"  {stem}_diff_amplified.png (10x amplified)")
    print(f"  {stem}_comparison.png (BBDM | VQGAN | Diff)")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
