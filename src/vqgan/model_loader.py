"""VQGAN Model Loader - handles checkpoint resolution and loading (Ultralytics-style).

Checkpoints are resolved by name and stored in a single directory.
If not found locally, they are automatically downloaded from HuggingFace.

Usage:
    loader = VQGANCheckpointLoader()
    path = loader.resolve("vqgan_f4.ckpt")  # Downloads if needed
    
    # Or with convenience function:
    path = get_checkpoint("vqgan_f8.ckpt")
"""
import os
from pathlib import Path
from typing import Optional

import torch

from src.data.hf_downloader import HFResourceManager


class VQGANCheckpointLoader:
    """
    Resolves and loads VQGAN checkpoints (Ultralytics-style).
    
    - Specify just the checkpoint filename (e.g., "vqgan_f4.ckpt")
    - Checkpoints are stored in CHECKPOINTS_DIR
    - Auto-downloads from HuggingFace if not found locally
    - Also supports full paths for flexibility
    """
    
    # Hardcoded local directory for all checkpoints
    CHECKPOINTS_DIR = Path("checkpoints")
    
    def __init__(
        self,
        hf_manager: Optional[HFResourceManager] = None,
    ):
        """
        Args:
            hf_manager: HuggingFace resource manager (creates default if None)
        """
        self.hf_manager = hf_manager or HFResourceManager()
    
    def resolve(
        self,
        ckpt: Optional[str] = None,
        auto_download: bool = True,
    ) -> Optional[Path]:
        """
        Resolve checkpoint by name or path.
        
        Args:
            ckpt: Checkpoint name (e.g., "vqgan_f4.ckpt") or full path.
                  If None, returns None (train from scratch).
            auto_download: If True, download from HF if not found locally
            
        Returns:
            Path to checkpoint, or None if ckpt is None
            
        Raises:
            FileNotFoundError: If checkpoint not found and can't be downloaded
        """
        if ckpt is None:
            print("No checkpoint specified, training from scratch")
            return None
        
        # Check if it's a full path (contains path separator)
        if os.path.sep in ckpt or '/' in ckpt:
            path = Path(ckpt)
            if path.exists():
                print(f"Using checkpoint: {path}")
                return path
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        
        # It's just a filename - look in CHECKPOINTS_DIR
        local_path = self.CHECKPOINTS_DIR / ckpt
        
        # Check if exists locally
        if local_path.exists():
            print(f"Using local checkpoint: {local_path}")
            return local_path
        
        # Try to download from HuggingFace
        if auto_download:
            print(f"Checkpoint '{ckpt}' not found locally, downloading from HuggingFace...")
            try:
                downloaded_path = self.hf_manager.get_model_checkpoint(
                    filename=ckpt,
                    local_dir=self.CHECKPOINTS_DIR,
                )
                return downloaded_path
            except Exception as e:
                raise FileNotFoundError(
                    f"Checkpoint '{ckpt}' not found locally at {local_path} "
                    f"and failed to download from HuggingFace: {e}"
                )
        
        raise FileNotFoundError(
            f"Checkpoint '{ckpt}' not found at {local_path}. "
            f"Set auto_download=True to download from HuggingFace."
        )
    
    def load_checkpoint(
        self,
        ckpt: Optional[str] = None,
        auto_download: bool = True,
        map_location: str = "cpu",
    ) -> Optional[dict]:
        """
        Load checkpoint, downloading from HF if needed.
        
        Args:
            ckpt: Checkpoint name or path
            auto_download: If True, download from HF if not found
            map_location: Device to load checkpoint to
            
        Returns:
            Loaded checkpoint dict, or None if ckpt is None
        """
        path = self.resolve(ckpt, auto_download)
        if path is None:
            return None
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=map_location)
        return checkpoint
    
    def get_state_dict(
        self,
        ckpt: Optional[str] = None,
        auto_download: bool = True,
    ) -> Optional[dict]:
        """
        Get state_dict from checkpoint.
        
        Returns:
            Model state dict, or None if ckpt is None
        """
        checkpoint = self.load_checkpoint(ckpt, auto_download)
        if checkpoint is None:
            return None
        
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint


# Convenience function
def get_checkpoint(
    ckpt: Optional[str] = None,
    auto_download: bool = True,
) -> Optional[Path]:
    """
    Get checkpoint path by name, downloading if needed.
    
    Args:
        ckpt: Checkpoint name (e.g., "vqgan_f4.ckpt") or full path, or None
        auto_download: Download from HF if not found locally
        
    Returns:
        Path to checkpoint file, or None if ckpt is None
        
    Examples:
        get_checkpoint("vqgan_f4.ckpt")      # -> checkpoints/vqgan_f4.ckpt
        get_checkpoint("vqgan_f8.ckpt")      # -> checkpoints/vqgan_f8.ckpt
        get_checkpoint("/full/path/model.ckpt")  # -> /full/path/model.ckpt
        get_checkpoint(None)                  # -> None (train from scratch)
    """
    loader = VQGANCheckpointLoader()
    return loader.resolve(ckpt, auto_download)
