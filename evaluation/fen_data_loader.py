"""
Data loader for FEN-labeled chess images.
Loads CSV with FEN labels and couples them with image paths.
"""

import os
import pandas as pd
import cv2
import numpy as np
from typing import Iterator, Tuple, List


class FenDataLoader:
    """Loads images paired with their FEN labels from a CSV file."""
    
    def __init__(self, csv_path: str, images_dir: str):
        assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"
        assert os.path.isdir(images_dir), f"Images dir not found: {images_dir}"
        
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        assert 'FEN' in df.columns, f"Missing 'FEN' column. Found: {df.columns.tolist()}"
        assert 'IMG_NAME' in df.columns, f"Missing 'IMG_NAME' column. Found: {df.columns.tolist()}"
        
        # Build image paths and filter missing
        df = df[['FEN', 'IMG_NAME']].copy()
        df['image_path'] = df['IMG_NAME'].apply(lambda x: os.path.join(images_dir, str(x)))
        
        missing_mask = ~df['image_path'].apply(os.path.isfile)
        n_missing = missing_mask.sum()
        if n_missing > 0:
            missing_names = df.loc[missing_mask, 'IMG_NAME'].tolist()
            print(f"WARNING: {n_missing} images not found in {images_dir}. Skipping: {missing_names[:5]}{'...' if n_missing > 5 else ''}")
            df = df[~missing_mask]
        
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        print(f"FenDataLoader: {len(self.df)} valid samples from {csv_path}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __iter__(self) -> Iterator[Tuple[str, str, np.ndarray]]:
        """Yields (image_path, fen, image) for each sample."""
        for _, row in self.df.iterrows():
            image = cv2.imread(row['image_path'])
            yield row['image_path'], row['FEN'], image
    
    def iter_batches(self, batch_size: int) -> Iterator[Tuple[List[str], List[str], List[np.ndarray]]]:
        """Yields batches of (image_paths, fens, images)."""
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
    
    def get_item(self, idx: int) -> Tuple[str, str, np.ndarray]:
        """Get single item by index."""
        row = self.df.iloc[idx]
        image = cv2.imread(row['image_path'])
        return row['image_path'], row['FEN'], image
