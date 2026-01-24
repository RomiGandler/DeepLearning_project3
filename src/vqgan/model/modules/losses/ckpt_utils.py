"""Checkpoint download utilities for LPIPS loss."""
import os
import hashlib

import requests
from tqdm import tqdm


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def _download(url: str, local_path: str, chunk_size: int = 1024):
    """Download a file from URL with progress bar."""
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))


def _md5_hash(path: str) -> str:
    """Compute MD5 hash of a file."""
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False) -> str:
    """
    Download LPIPS checkpoint if needed and return path.
    
    Args:
        name: Checkpoint name (e.g., "vgg_lpips")
        root: Directory to store checkpoints (relative to this file or absolute)
        check: Whether to verify MD5 hash
        
    Returns:
        Path to the checkpoint file
    """
    assert name in URL_MAP, f"Unknown checkpoint: {name}. Available: {list(URL_MAP.keys())}"
    
    # Convert to absolute path relative to this file if not absolute
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(__file__), root)
    
    path = os.path.join(root, CKPT_MAP[name])
    
    if not os.path.exists(path) or (check and _md5_hash(path) != MD5_MAP[name]):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
        _download(URL_MAP[name], path)
        md5 = _md5_hash(path)
        assert md5 == MD5_MAP[name], f"MD5 mismatch: expected {MD5_MAP[name]}, got {md5}"
    
    return path

