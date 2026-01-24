"""HuggingFace resource manager for chess model - handles both dataset and model downloads."""
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, hf_hub_download


class HFResourceManager:
    """Downloads and caches resources from HuggingFace Hub."""
    
    # Repositories
    DATA_REPO_ID = "roni-hershko/chess_data"
    MODEL_REPO_ID = "roni-hershko/chess_model"
    
    def __init__(
        self,
        local_cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Args:
            local_cache_dir: Where to cache downloaded resources
            token: HuggingFace token for private repos (or set HF_TOKEN env var)
        """
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
        
        print(f"Downloading dataset from {self.DATA_REPO_ID}...")
        
        # Download entire repo for dataset (contains train/val/test folders)
        local_path = snapshot_download(
            repo_id=self.DATA_REPO_ID,
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
        filename: str,
        local_dir: Optional[Path] = None,
        force_download: bool = False,
    ) -> Path:
        """
        Get a model checkpoint file, downloading if necessary.
        
        Args:
            filename: Checkpoint filename in the repo (e.g., "vqgan_f4.ckpt", "bbdm_chess.pth")
            local_dir: Where to save the file (default: internal cache)
            force_download: If True, re-download even if cached
            
        Returns:
            Path to the local checkpoint file
        """
        if not filename:
            raise ValueError("Filename must be provided for model checkpoint download")

        # Use provided local_dir or default to internal cache
        if local_dir is None:
            local_dir = self.local_cache_dir / "models"
        local_dir = Path(local_dir)
        
        local_path = local_dir / filename
        
        if not force_download and local_path.exists():
            print(f"Model checkpoint already cached at {local_path}")
            return local_path
        
        print(f"Downloading {filename} from {self.MODEL_REPO_ID}...")
        
        # Download specific file
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=self.MODEL_REPO_ID,
            filename=filename,
            repo_type="dataset", # Models are stored in a dataset repo in this setup
            local_dir=str(local_dir),
            token=self.token,
        )
        
        print(f"Model checkpoint downloaded to {downloaded_path}")
        return Path(downloaded_path)
