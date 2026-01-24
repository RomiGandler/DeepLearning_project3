"""HuggingFace resource manager for chess model - handles both dataset and model downloads."""
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, hf_hub_download


class HFResourceManager:
    """Downloads and caches resources from HuggingFace Hub."""
    
    # Default repository and paths
    DEFAULT_REPO_ID = "roni-hershko/chess_model"
    DEFAULT_VQGAN_FILENAME = "vqgan_f8.ckpt"  # f8 = 256/8 = 32 latent size
    DEFAULT_BBDM_FILENAME = "bbdm_chess.pth"  # Default BBDM checkpoint
    
    def __init__(
        self,
        repo_id: str = None,
        local_cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Args:
            repo_id: HuggingFace repo ID (default: roni-hershko/chess_model)
            local_cache_dir: Where to cache downloaded resources
            token: HuggingFace token for private repos (or set HF_TOKEN env var)
        """
        self.repo_id = repo_id or self.DEFAULT_REPO_ID
        
        if local_cache_dir is None:
            local_cache_dir = os.path.join(os.path.dirname(__file__), "hf_cache")
        
        self.local_cache_dir = Path(local_cache_dir)
        self.token = token
    
    # =========================================================================
    # Dataset Downloads
    # =========================================================================
    
    def get_dataset(self, force_download: bool = False) -> Path:
        """
        Get the chess dataset, downloading if necessary.
        
        Args:
            force_download: If True, re-download even if cached
            
        Returns:
            Path to the dataset directory
        """
        dataset_dir = self.local_cache_dir / "dataset"
        
        if not force_download and dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"Dataset already cached at {dataset_dir}")
            return dataset_dir
        
        print(f"Downloading dataset from {self.repo_id}...")
        
        # Download entire repo for dataset (contains train/val/test folders)
        local_path = snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            token=self.token,
            ignore_patterns=["*.ckpt", "*.pth", "*.pt"],  # Skip model files
        )
        
        print(f"Dataset downloaded to {local_path}")
        return Path(local_path)
    
    # =========================================================================
    # Model Downloads
    # =========================================================================
    
    def get_model_checkpoint(
        self,
        filename: str = None,
        local_dir: Optional[Path] = None,
        force_download: bool = False,
    ) -> Path:
        """
        Get a model checkpoint file, downloading if necessary.
        
        Args:
            filename: Checkpoint filename in the repo (e.g., "vqgan_f4.ckpt", "vqgan_f8.ckpt")
            local_dir: Where to save the file (default: internal cache)
            force_download: If True, re-download even if cached
            
        Returns:
            Path to the local checkpoint file
        """
        filename = filename or self.DEFAULT_VQGAN_FILENAME
        
        # Use provided local_dir or default to internal cache
        if local_dir is None:
            local_dir = self.local_cache_dir / "models"
        local_dir = Path(local_dir)
        
        local_path = local_dir / filename
        
        if not force_download and local_path.exists():
            print(f"Model checkpoint already cached at {local_path}")
            return local_path
        
        print(f"Downloading {filename} from {self.repo_id}...")
        
        # Download specific file
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=self.token,
        )
        
        print(f"Model checkpoint downloaded to {downloaded_path}")
        return Path(downloaded_path)
    
    def get_vqgan_checkpoint(
        self,
        local_dir: Optional[Path] = None,
        force_download: bool = False,
    ) -> Path:
        """Convenience method to get the default VQGAN checkpoint."""
        return self.get_model_checkpoint(
            filename=self.DEFAULT_VQGAN_FILENAME,
            local_dir=local_dir,
            force_download=force_download,
        )
    
    def get_bbdm_checkpoint(
        self,
        local_dir: Optional[Path] = None,
        force_download: bool = False,
    ) -> Path:
        """Convenience method to get the default BBDM checkpoint."""
        return self.get_model_checkpoint(
            filename=self.DEFAULT_BBDM_FILENAME,
            local_dir=local_dir,
            force_download=force_download,
        )
    
    @classmethod
    def resolve_checkpoint_path(
        cls,
        ckpt_path: Optional[str],
        checkpoint_type: str = "vqgan",
    ) -> Optional[str]:
        """
        Resolve a checkpoint path, downloading from HuggingFace if needed.
        
        Args:
            ckpt_path: Local path to checkpoint, or None to auto-download
            checkpoint_type: One of "vqgan" or "bbdm"
            
        Returns:
            Resolved path to checkpoint file, or None if no checkpoint needed
        """
        # If path provided and exists, use it directly
        if ckpt_path is not None:
            path = Path(ckpt_path)
            if path.exists():
                return str(path)
            else:
                print(f"Checkpoint not found at {ckpt_path}, downloading from HuggingFace...")
        else:
            print(f"No {checkpoint_type} checkpoint path provided, downloading from HuggingFace...")
        
        # Download from HuggingFace
        hf_manager = cls()
        if checkpoint_type == "vqgan":
            downloaded_path = hf_manager.get_vqgan_checkpoint()
        elif checkpoint_type == "bbdm":
            downloaded_path = hf_manager.get_bbdm_checkpoint()
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
        
        return str(downloaded_path)
