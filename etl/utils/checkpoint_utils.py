"""
Checkpoint utilities for ETL pipelines.

Handles automatic download of model checkpoints from HuggingFace.
"""

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


# HuggingFace repository containing model checkpoints
MODEL_REPO_ID = "roni-hershko/chess_model"

# SAM model filename
SAM_MODEL_FILENAME = "sam3.pt"

# Default local directory for checkpoints
DEFAULT_CHECKPOINTS_DIR = Path("checkpoints")


def get_sam_checkpoint(
    checkpoints_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """
    Get SAM checkpoint path, downloading from HuggingFace if needed.

    Args:
        checkpoints_dir: Directory to store checkpoints (default: ./checkpoints)
        force_download: If True, re-download even if exists locally

    Returns:
        Path to sam3.pt checkpoint file
    """
    checkpoints_dir = checkpoints_dir or DEFAULT_CHECKPOINTS_DIR
    local_path = checkpoints_dir / SAM_MODEL_FILENAME

    # Check if exists locally
    if not force_download and local_path.exists():
        print(f"Using local SAM model: {local_path}")
        return local_path

    # Download from HuggingFace
    print(f"SAM model not found locally, downloading from HuggingFace...")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=SAM_MODEL_FILENAME,
        repo_type="model",
        local_dir=str(checkpoints_dir),
    )

    print(f"SAM model downloaded to {downloaded_path}")
    return Path(downloaded_path)
