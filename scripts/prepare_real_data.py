import os
import csv
import shutil
import json

def prepare_real_data():
    base_dir = "Labeled Chess data (PGN games will be added later)-20251211"
    output_dir = "dataset/trainB"
    metadata_file = "dataset/metadata.json"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    metadata = []
    
    # Iterate through each game directory
    for game_dir in os.listdir(base_dir):
        game_path = os.path.join(base_dir, game_dir)
        if not os.path.isdir(game_path):
            continue
            
        # Find the CSV file
        csv_file = None
        for f in os.listdir(game_path):
            if f.endswith(".csv"):
                csv_file = os.path.join(game_path, f)
                break
        
        if not csv_file:
            print(f"No CSV found in {game_path}")
            continue
            
        tagged_images_path = os.path.join(game_path, "tagged_images")
        if not os.path.exists(tagged_images_path):
            print(f"No tagged_images folder in {game_path}")
            continue
            
        print(f"Processing {game_dir}...")
        
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_num = row['from_frame']
                fen = row['fen']
                
                # Format frame number to match filename (e.g., 200 -> frame_000200.jpg)
                frame_filename = f"frame_{int(frame_num):06d}.jpg"
                src_path = os.path.join(tagged_images_path, frame_filename)
                
                if os.path.exists(src_path):
                    dest_filename = f"{game_dir}_{frame_filename}"
                    dest_path = os.path.join(output_dir, dest_filename)
                    
                    shutil.copy2(src_path, dest_path)
                    
                    metadata.append({
                        "image_path": dest_filename,
                        "fen": fen,
                        "original_game": game_dir,
                        "original_frame": frame_num
                    })
                else:
                    # Some frames might not be in the tagged_images if they are outside the set
                    pass

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Finished! Processed {len(metadata)} images.")

if __name__ == "__main__":
    prepare_real_data()

