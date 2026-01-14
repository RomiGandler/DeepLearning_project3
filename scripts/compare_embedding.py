#!/usr/bin/env python3
"""
Script to compare latent embeddings between synthetic and real chess board images.

Creates a 2x4 grid visualization:
- Row 1 (Real):      Input image | Downsampled image | Downsampled mask | Embedding Ch0
- Row 2 (Synthetic): Input image | Downsampled image | Downsampled mask | Embedding Ch0
"""

import sys
import os

# Add taming-transformers to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'taming-transformers'))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from omegaconf import OmegaConf

from taming.models.vqgan import VQModel
from main import instantiate_from_config


# ============================================================================
# PLACEHOLDERS - Update these paths before running
# ============================================================================

# Path to your finetuned VQGAN checkpoint
CHECKPOINT_PATH = "/home/avinoamd/roni/taming-transformers/logs/2026-01-05T16-08-17_chess_finetune/checkpoints/last.ckpt"

# Path to the config file used for training
CONFIG_PATH = "/home/avinoamd/roni/taming-transformers/configs/chess_finetune.yaml"

# Path to real chess board image
REAL_IMAGE_PATH = "/home/avinoamd/roni/BBDM/friefeld_data/test/A/g2_1280.jpg"

# Path to real chess pieces mask
REAL_MASK_PATH = "/home/avinoamd/roni/BBDM/friefeld_data/test/masks/g2_1280.png"

# Path to synthetic chess board image
SYNTHETIC_IMAGE_PATH = "/home/avinoamd/roni/BBDM/friefeld_data/test/B/g2_1280.jpg"

# Path to synthetic chess pieces mask
SYNTHETIC_MASK_PATH = "/home/avinoamd/roni/BBDM/friefeld_data/test/masks/g2_1280.png"

# Output path for the grid image
OUTPUT_PATH = "/home/avinoamd/roni/scripts/embedding_comparison.png"

# ============================================================================


def load_model(config_path: str, checkpoint_path: str, device: str = "cuda") -> VQModel:
    """Load VQGAN model from config and checkpoint."""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    
    # Load checkpoint
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def load_and_preprocess_image(image_path: str, size: int = 256) -> torch.Tensor:
    """Load and preprocess image for VQGAN."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def load_mask(mask_path: str, size: int = 256) -> np.ndarray:
    """Load mask image and resize to specified size."""
    mask = Image.open(mask_path).convert("L")  # Grayscale
    mask = mask.resize((size, size), Image.NEAREST)
    mask_np = np.array(mask) / 255.0  # Normalize to [0, 1]
    return mask_np


def downsample_image(image_np: np.ndarray, target_size: int) -> np.ndarray:
    """Downsample image to target size using PIL."""
    # image_np is (H, W, 3) in range [0, 1]
    img_uint8 = (image_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_downsampled = img_pil.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img_downsampled) / 255.0


def downsample_mask(mask_np: np.ndarray, target_size: int) -> np.ndarray:
    """Downsample mask to target size using nearest neighbor."""
    # mask_np is (H, W) in range [0, 1]
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8, mode='L')
    mask_downsampled = mask_pil.resize((target_size, target_size), Image.NEAREST)
    return np.array(mask_downsampled) / 255.0


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor [-1, 1] back to numpy image [0, 1]."""
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = np.clip(img, 0, 1)
    return img


def get_embedding_channel(embedding: torch.Tensor, channel_idx: int = 0) -> np.ndarray:
    """
    Extract a single channel from embedding tensor.
    
    Args:
        embedding: Tensor of shape (1, C, H, W)
        channel_idx: Which channel to extract
        
    Returns:
        Numpy array of shape (H, W) normalized to [0, 1]
    """
    emb = embedding.squeeze(0).cpu().numpy()  # (C, H, W)
    channel = emb[channel_idx]  # (H, W)
    # Normalize to [0, 1]
    channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    return channel


def create_comparison_grid(
    real_image: np.ndarray,
    real_image_down: np.ndarray,
    real_mask_down: np.ndarray,
    real_emb_ch0: np.ndarray,
    synthetic_image: np.ndarray,
    synthetic_image_down: np.ndarray,
    synthetic_mask_down: np.ndarray,
    synthetic_emb_ch0: np.ndarray,
    output_path: str
) -> None:
    """
    Create and save a 2x4 grid comparison image.
    
    Layout:
    Row 1 (Real):      Input | Downsampled | Mask (down) | Embedding Ch0
    Row 2 (Synthetic): Input | Downsampled | Mask (down) | Embedding Ch0
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    # Column titles
    col_titles = ["Input Image", "Downsampled Image", "Downsampled Mask", "Embedding Ch0"]
    
    # Row 1: Real
    axes[0, 0].imshow(real_image)
    axes[0, 0].set_title(col_titles[0], fontsize=11)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(real_image_down)
    axes[0, 1].set_title(col_titles[1], fontsize=11)
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(real_mask_down, cmap="gray")
    axes[0, 2].set_title(col_titles[2], fontsize=11)
    axes[0, 2].axis("off")
    
    axes[0, 3].imshow(real_emb_ch0, cmap="viridis")
    axes[0, 3].set_title(col_titles[3], fontsize=11)
    axes[0, 3].axis("off")
    
    # Row 2: Synthetic
    axes[1, 0].imshow(synthetic_image)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(synthetic_image_down)
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(synthetic_mask_down, cmap="gray")
    axes[1, 2].axis("off")
    
    axes[1, 3].imshow(synthetic_emb_ch0, cmap="viridis")
    axes[1, 3].axis("off")
    
    # Add row labels
    fig.text(0.01, 0.72, "Real", fontsize=14, fontweight="bold", rotation=90, va="center")
    fig.text(0.01, 0.28, "Synthetic", fontsize=14, fontweight="bold", rotation=90, va="center")
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.04)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison grid to {output_path}")


def main():
    """Main function to run the embedding comparison."""
    # Check if placeholder paths have been updated
    if "/path/to/" in CHECKPOINT_PATH:
        print("ERROR: Please update CHECKPOINT_PATH in the script with your actual checkpoint path.")
        sys.exit(1)
    if "/path/to/" in REAL_IMAGE_PATH:
        print("ERROR: Please update REAL_IMAGE_PATH in the script with your actual image path.")
        sys.exit(1)
    if "/path/to/" in SYNTHETIC_IMAGE_PATH:
        print("ERROR: Please update SYNTHETIC_IMAGE_PATH in the script with your actual image path.")
        sys.exit(1)
    if "/path/to/" in REAL_MASK_PATH:
        print("ERROR: Please update REAL_MASK_PATH in the script with your actual mask path.")
        sys.exit(1)
    if "/path/to/" in SYNTHETIC_MASK_PATH:
        print("ERROR: Please update SYNTHETIC_MASK_PATH in the script with your actual mask path.")
        sys.exit(1)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device)
    
    # Load images
    print(f"Loading real image from: {REAL_IMAGE_PATH}")
    real_tensor = load_and_preprocess_image(REAL_IMAGE_PATH).to(device)
    
    print(f"Loading synthetic image from: {SYNTHETIC_IMAGE_PATH}")
    synthetic_tensor = load_and_preprocess_image(SYNTHETIC_IMAGE_PATH).to(device)
    
    # Load masks (at 256x256)
    print(f"Loading real mask from: {REAL_MASK_PATH}")
    real_mask = load_mask(REAL_MASK_PATH, size=256)
    
    print(f"Loading synthetic mask from: {SYNTHETIC_MASK_PATH}")
    synthetic_mask = load_mask(SYNTHETIC_MASK_PATH, size=256)
    
    # Get embeddings using forward_with_embedding
    print("Computing embeddings...")
    with torch.no_grad():
        real_recon, real_quant, _ = model.forward_with_embedding(real_tensor)
        synthetic_recon, synthetic_quant, _ = model.forward_with_embedding(synthetic_tensor)
    
    # Get embedding spatial size (should be 16 for f16 VQGAN with 256 input)
    embedding_size = real_quant.shape[-1]
    print(f"Embedding shape: {real_quant.shape} (spatial size: {embedding_size})")
    
    # Convert tensors to numpy images
    real_image_np = tensor_to_numpy_image(real_tensor)
    synthetic_image_np = tensor_to_numpy_image(synthetic_tensor)
    
    # Downsample images to embedding size
    real_image_down = downsample_image(real_image_np, embedding_size)
    synthetic_image_down = downsample_image(synthetic_image_np, embedding_size)
    
    # Downsample masks to embedding size
    real_mask_down = downsample_mask(real_mask, embedding_size)
    synthetic_mask_down = downsample_mask(synthetic_mask, embedding_size)
    
    # Extract channel 0 from embeddings
    real_emb_ch0 = get_embedding_channel(real_quant, channel_idx=0)
    synthetic_emb_ch0 = get_embedding_channel(synthetic_quant, channel_idx=0)
    
    # Create and save comparison grid
    create_comparison_grid(
        real_image_np,
        real_image_down,
        real_mask_down,
        real_emb_ch0,
        synthetic_image_np,
        synthetic_image_down,
        synthetic_mask_down,
        synthetic_emb_ch0,
        OUTPUT_PATH
    )
    
    # Print some statistics
    print("\n--- Embedding Statistics ---")
    print(f"Real embedding - min: {real_quant.min().item():.4f}, max: {real_quant.max().item():.4f}, "
          f"mean: {real_quant.mean().item():.4f}, std: {real_quant.std().item():.4f}")
    print(f"Synthetic embedding - min: {synthetic_quant.min().item():.4f}, max: {synthetic_quant.max().item():.4f}, "
          f"mean: {synthetic_quant.mean().item():.4f}, std: {synthetic_quant.std().item():.4f}")
    
    # Compute cosine similarity between embeddings
    real_flat = real_quant.flatten()
    synthetic_flat = synthetic_quant.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(real_flat.unsqueeze(0), synthetic_flat.unsqueeze(0))
    print(f"Cosine similarity between embeddings: {cosine_sim.item():.4f}")
    
    # Compute L2 distance
    l2_dist = torch.norm(real_flat - synthetic_flat).item()
    print(f"L2 distance between embeddings: {l2_dist:.4f}")


if __name__ == "__main__":
    main()
