import os
import shutil
import csv
from tqdm import tqdm

# --- CONFIGURATION ---

# 1. Path to the NEW massive folders (Game 8-13)
RAW_DATA_DIR = "pgn_data"

# 2. Path to the OLD labeled folders (Game 2,4,5,6,7)
# Updated to match your actual folder name:
LABELED_DATA_DIR = "labeled_chess_data" 

# Destination
DEST_DIR = "dataset/trainB"

# Sampling rate for the MASSIVE folders only (1 every 50)
STEP = 50

def collect_all_real_data():
    # Ensure destination exists
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Destination folder ready: {DEST_DIR}")
    
    total_copied = 0

    # --- PART 1: Process the LABELED Data (Copy ALL via CSV) ---
    print(f"\n--- Processing Labeled Data from: {LABELED_DATA_DIR} ---")
    
    if os.path.exists(LABELED_DATA_DIR):
        # Iterate over game2_per_frame, game5_per_frame, etc.
        for game_dir in sorted(os.listdir(LABELED_DATA_DIR)):
            game_path = os.path.join(LABELED_DATA_DIR, game_dir)
            
            if not os.path.isdir(game_path):
                continue
            
            # Find CSV file inside (e.g., game5.csv)
            csv_file = None
            for f in os.listdir(game_path):
                if f.endswith(".csv"):
                    csv_file = os.path.join(game_path, f)
                    break
            
            # Find 'tagged_images' folder
            tagged_path = os.path.join(game_path, "tagged_images")
            
            if csv_file and os.path.exists(tagged_path):
                print(f"Collecting from {game_dir}...")
                
                try:
                    with open(csv_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # The CSV has 'from_frame'
                            frame_num = row.get('from_frame')
                            if not frame_num: 
                                continue
                            
                            # Construct filename: frame_000044.jpg
                            fname = f"frame_{int(frame_num):06d}.jpg"
                            src = os.path.join(tagged_path, fname)
                            
                            if os.path.exists(src):
                                # Create unique name for trainB: game5_per_frame_frame_000044.jpg
                                dst_name = f"{game_dir}_{fname}"
                                dst_path = os.path.join(DEST_DIR, dst_name)
                                
                                shutil.copy2(src, dst_path)
                                total_copied += 1
                except Exception as e:
                    print(f"Error reading CSV in {game_dir}: {e}")
            else:
                # Just for debugging info
                pass
    else:
        print(f"Error: Directory '{LABELED_DATA_DIR}' not found. Please check name.")

    # --- PART 2: Process the RAW Data (Sampling) ---
    print(f"\n--- Processing Raw Data from: {RAW_DATA_DIR} ---")
    
    if os.path.exists(RAW_DATA_DIR):
        games = sorted(os.listdir(RAW_DATA_DIR))
        for game_folder in games:
            game_path = os.path.join(RAW_DATA_DIR, game_folder)
            
            if not os.path.isdir(game_path):
                continue
                
            images_path = os.path.join(game_path, "images")
            
            if os.path.exists(images_path):
                # Get all images
                all_imgs = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg','.png'))])
                total_in_game = len(all_imgs)
                
                if total_in_game > 0:
                    print(f"{game_folder}: Found {total_in_game} images. Sampling every {STEP}th...")
                    
                    # Take 1 image every 50 frames
                    for i in range(0, total_in_game, STEP):
                        img_name = all_imgs[i]
                        
                        # Unique name: game8_frame_0001.jpg
                        dst_name = f"{game_folder}_{img_name}"
                        src = os.path.join(images_path, img_name)
                        
                        shutil.copy2(src, os.path.join(DEST_DIR, dst_name))
                        total_copied += 1
    else:
        print(f"Warning: Directory '{RAW_DATA_DIR}' not found.")

    print("-" * 30)
    print(f"DONE! Total images in trainB: {total_copied}")

if __name__ == "__main__":
    collect_all_real_data()