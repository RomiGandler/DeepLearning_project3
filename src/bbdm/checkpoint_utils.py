"""Checkpoint resolution utilities for BBDM.

Provides a single source of truth for resolving checkpoint paths.
Both VQGAN and BBDM use the same logic:
    - None: Return None (no checkpoint)
    - "model.pth": Check local checkpoints/ dir, download from HF if missing
    - "/full/path/model.pth": Use directly
"""
from pathlib import Path
from typing import Optional

# Default directory for all checkpoints (VQGAN + BBDM)
CHECKPOINTS_DIR = Path("checkpoints")


def resolve_checkpoint(
    ckpt: Optional[str],
    checkpoints_dir: Path = CHECKPOINTS_DIR,
) -> Optional[Path]:
    """
    Resolve checkpoint by name or path, downloading from HF if needed.
    
    Args:
        ckpt: One of:
            - None: Return None (train from scratch / no checkpoint)
            - "model.pth": Check local dir, download from HF if missing
            - "/full/path/model.pth": Use directly
        checkpoints_dir: Where to look for / save checkpoints
            
    Returns:
        Path to checkpoint file, or None if ckpt is None
        
    Raises:
        FileNotFoundError: If full path doesn't exist or download fails
    """
    if ckpt is None:
        return None
    
    # Full path (contains path separator)
    if '/' in ckpt or '\\' in ckpt:
        path = Path(ckpt)
        if path.exists():
            print(f"Using checkpoint: {path}")
            return path
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    
    # Just a filename - check local first
    checkpoints_dir = Path(checkpoints_dir)
    local_path = checkpoints_dir / ckpt
    if local_path.exists():
        print(f"Using local checkpoint: {local_path}")
        return local_path
    
    # Download from HuggingFace
    print(f"Checkpoint '{ckpt}' not found locally, downloading from HuggingFace...")
    from src.data.hf_downloader import HFResourceManager
    hf_manager = HFResourceManager()
    try:
        downloaded = hf_manager.get_model_checkpoint(
            filename=ckpt,
            local_dir=checkpoints_dir,
        )
        return Path(downloaded)
    except Exception as e:
        raise FileNotFoundError(
            f"Checkpoint '{ckpt}' not found locally at {local_path} "
            f"and failed to download from HuggingFace: {e}"
        )
