"""
SAM Model Loader - handles checkpoint resolution and auto-download from HuggingFace.

The SAM model (sam3.pt) is stored in the checkpoints directory.
If not found locally, it is automatically downloaded from HuggingFace.

Usage:
    loader = SAMCheckpointLoader()
    path = loader.resolve()  # Downloads sam3.pt if needed
    
    # Or with convenience function:
    path = get_sam_checkpoint()
"""

from pathlib import Path
from typing import Optional

from src.data.hf_downloader import HFResourceManager


# SAM model filename (fixed - only one model available)
SAM_MODEL_FILENAME = "sam3.pt"

# Default checkpoint directory
CHECKPOINTS_DIR = Path("checkpoints")


class SAMCheckpointLoader:
    """
    Resolves and loads the SAM checkpoint (sam3.pt).
    
    - Checkpoints are stored in CHECKPOINTS_DIR
    - Auto-downloads from HuggingFace if not found locally
    """
    
    def __init__(
        self,
        hf_manager: Optional[HFResourceManager] = None,
        checkpoints_dir: Optional[Path] = None,
    ):
        """
        Args:
            hf_manager: HuggingFace resource manager (creates default if None)
            checkpoints_dir: Override default checkpoints directory
        """
        self.hf_manager = hf_manager or HFResourceManager()
        self.checkpoints_dir = checkpoints_dir or CHECKPOINTS_DIR
    
    def resolve(self) -> Path:
        """
        Resolve SAM checkpoint path, downloading from HuggingFace if needed.
        
        Returns:
            Path to sam3.pt checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint not found and can't be downloaded
        """
        local_path = self.checkpoints_dir / SAM_MODEL_FILENAME
        
        # Check if exists locally
        if local_path.exists():
            print(f"Using local SAM model: {local_path}")
            return local_path
        
        # Download from HuggingFace
        print(f"SAM model not found locally, downloading from HuggingFace...")
        try:
            downloaded_path = self.hf_manager.get_model_checkpoint(
                filename=SAM_MODEL_FILENAME,
                local_dir=self.checkpoints_dir,
            )
            return downloaded_path
        except Exception as e:
            raise FileNotFoundError(
                f"SAM model not found locally at {local_path} "
                f"and failed to download from HuggingFace: {e}"
            )


def get_sam_checkpoint() -> Path:
    """
    Get SAM checkpoint path, downloading from HuggingFace if needed.
    
    Returns:
        Path to sam3.pt checkpoint file
        
    Example:
        path = get_sam_checkpoint()  # -> checkpoints/sam3.pt
    """
    loader = SAMCheckpointLoader()
    return loader.resolve()
