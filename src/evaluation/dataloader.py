"""
Unified evaluation dataset for chess board images with FEN labels.

Replaces FenDataLoader while following BaseChessDataset conventions.
Supports both local paths and HuggingFace auto-download.
"""

import os
from pathlib import Path
from typing import Optional, Iterator, Tuple, List
import pandas as pd
import numpy as np
import cv2

from src.data.hf_downloader import HFResourceManager


class ChessEvalDataset:
    """
    Evaluation dataset that loads images + FEN labels from gt.csv.
    
    Directory structure expected:
        dataset_path/
        ├── train/
        │   ├── A/          # Condition images (not used for eval)
        │   ├── B/          # Target images (used if generated_dir=None)
        │   └── gt.csv      # FEN, IMG_NAME, GAME columns
        ├── val/
        └── test/
    
    Usage:
        # Evaluate original B/ images (sanity check)
        dataset = ChessEvalDataset(dataset_path="path/to/data", stage="test")
        
        # Evaluate model-generated images
        dataset = ChessEvalDataset(
            dataset_path="path/to/data",
            stage="test", 
            generated_dir="outputs/model_results/"
        )
        
        # Auto-download from HuggingFace
        dataset = ChessEvalDataset(dataset_path=None, stage="test")
        
        # Iterate (FenDataLoader-compatible)
        for image_path, fen, image in dataset:
            ...
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        stage: str = 'test',
        generated_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_path: Root dataset path. If None, downloads from HuggingFace.
            stage: Split to use ('train', 'val', 'test')
            generated_dir: Path to model outputs. If None, uses B/ from dataset.
        """
        # Resolve dataset path (same pattern as BaseChessDataset)
        if dataset_path is None:
            hf_manager = HFResourceManager()
            dataset_path = hf_manager.get_dataset()
        
        self.dataset_path = Path(dataset_path)
        self.stage = stage
        self.stage_path = self.dataset_path / stage
        
        assert self.stage_path.exists(), f"Stage directory not found: {self.stage_path}"
        
        # Load FEN metadata from gt.csv
        csv_path = self.stage_path / 'gt.csv'
        assert csv_path.exists(), f"gt.csv not found at {csv_path}"
        
        self.df = pd.read_csv(csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]
        
        assert 'FEN' in self.df.columns, f"Missing 'FEN' column in {csv_path}"
        assert 'IMG_NAME' in self.df.columns, f"Missing 'IMG_NAME' column in {csv_path}"
        
        # Determine image source
        if generated_dir is not None:
            self.images_dir = Path(generated_dir)
        else:
            self.images_dir = self.stage_path / 'B'
        
        assert self.images_dir.exists(), f"Images directory not found: {self.images_dir}"
        
        # Build image paths and filter missing
        self.df['image_path'] = self.df['IMG_NAME'].apply(
            lambda x: str(self.images_dir / x)
        )
        
        exists_mask = self.df['image_path'].apply(os.path.isfile)
        n_missing = (~exists_mask).sum()
        if n_missing > 0:
            missing = self.df.loc[~exists_mask, 'IMG_NAME'].tolist()
            print(f"Warning: {n_missing} images not found, skipping: "
                  f"{missing[:5]}{'...' if n_missing > 5 else ''}")
        
        self.df = self.df[exists_mask].reset_index(drop=True)
        
        print(f"ChessEvalDataset: {len(self.df)} samples from {stage}/ "
              f"(images: {self.images_dir})")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[str, str, np.ndarray]:
        """Get single sample as (image_path, fen, image_bgr)."""
        row = self.df.iloc[idx]
        image = cv2.imread(row['image_path'])
        return row['image_path'], row['FEN'], image
    
    def __iter__(self) -> Iterator[Tuple[str, str, np.ndarray]]:
        """Iterate yielding (image_path, fen, image_bgr) tuples."""
        for idx in range(len(self)):
            yield self[idx]
    
    def iter_batches(self, batch_size: int) -> Iterator[Tuple[List[str], List[str], List[np.ndarray]]]:
        """Yield batches of (paths, fens, images)."""
        paths, fens, images = [], [], []
        for path, fen, img in self:
            paths.append(path)
            fens.append(fen)
            images.append(img)
            if len(paths) == batch_size:
                yield paths, fens, images
                paths, fens, images = [], [], []
        if paths:
            yield paths, fens, images
    
    def get_fen(self, filename: str) -> Optional[str]:
        """Look up FEN by image filename."""
        match = self.df[self.df['IMG_NAME'] == filename]
        return match['FEN'].iloc[0] if len(match) > 0 else None
