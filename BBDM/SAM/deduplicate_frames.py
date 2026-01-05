import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import imagehash
from PIL import Image

def deduplicate_frames(images_dir, mse_threshold=300, hash_threshold=15, dry_run=True):
    """
    Remove duplicates using BOTH:
    - Downscaled MSE comparison
    - Perceptual hashing
    
    If EITHER method says images are similar -> duplicate
    """
    images_dir = Path(images_dir)
    duplicates_dir = images_dir.parent / "duplicates"
    
    image_files = list(images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images")
    
    if len(image_files) < 2:
        print("Not enough images to compare.")
        return
    
    def get_small(img_path):
        """Load and downscale image for MSE comparison."""
        img = cv2.imread(str(img_path))
        return cv2.resize(img, (64, 64))
    
    def calc_mse(small1, small2):
        """Calculate MSE between two small images."""
        err = np.sum((small1.astype("float") - small2.astype("float")) ** 2)
        return err / (64 * 64 * 3)
    
    def get_phash(img_path):
        """Get perceptual hash of image."""
        return imagehash.phash(Image.open(str(img_path)))
    
    duplicates = []
    kept = [image_files[0]]
    
    # Cache for previous image data
    prev_small = get_small(image_files[0])
    prev_hash = get_phash(image_files[0])
    
    for i, img_path in enumerate(image_files[1:], 1):
        curr_small = get_small(img_path)
        curr_hash = get_phash(img_path)
        
        # Calculate both metrics
        mse = calc_mse(prev_small, curr_small)
        hash_diff = abs(prev_hash - curr_hash)
        
        # Duplicate if EITHER method says so
        is_duplicate = (mse < mse_threshold) or (hash_diff < hash_threshold)
        
        if is_duplicate:
            duplicates.append(img_path)
        else:
            kept.append(img_path)
            prev_small = curr_small
            prev_hash = curr_hash
        
        if i % 500 == 0:
            print(f"  Processed {i}/{len(image_files)} | Kept: {len(kept)} | Dups: {len(duplicates)}")
    
    print(f"\n=== Summary ===")
    print(f"Total: {len(image_files)}")
    print(f"Unique: {len(kept)}")
    print(f"Duplicates: {len(duplicates)}")
    
    if dry_run:
        print("\n[DRY RUN] Use --execute to actually move files.")
    else:
        os.makedirs(duplicates_dir, exist_ok=True)
        for dup_path in duplicates:
            shutil.move(str(dup_path), str(duplicates_dir / dup_path.name))
        print(f"\nMoved {len(duplicates)} duplicates to: {duplicates_dir}")

if __name__ == "__main__":
    import sys
    
    images_dir = "/home/avinoamd/roni/BBDM/pgn_data/game11-20260104T231534Z-3-001/game11/images"
    
    dry_run = "--execute" not in sys.argv
    
    if dry_run:
        print("=== DRY RUN (use --execute to actually move files) ===\n")
    
    # Adjust thresholds to get ~150 unique frames from 11,000
    # mse_threshold: higher = more aggressive (try 500-2000)
    # hash_threshold: higher = more aggressive (try 20-35)
    deduplicate_frames(images_dir, mse_threshold=1500, hash_threshold=30, dry_run=dry_run)
