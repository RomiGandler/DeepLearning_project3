import os
import json
import subprocess
from tqdm import tqdm

def generate_synthetic_data():
    metadata_file = "dataset/metadata.json"
    output_dir = "dataset/trainA"
    blender_path = "/Applications/Blender.app/Contents/MacOS/Blender"
    blend_file = "blender/chess-set.blend"
    script_file = "blender/chess_position_api_v2.py"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found. Run prepare_real_data.py first.")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
    print(f"Starting generation of {len(metadata)} synthetic images...")
    
    # We will use the overhead view primarily as it matches the real data samples
    # The script generates 3 images: 1_overhead, 2_east, 3_west (or 2_west, 3_east)
    # We will rename the 1_overhead.png to match the real image name
    
    for entry in tqdm(metadata):
        image_name = entry['image_path']
        fen = entry['fen']
        
        # Output filename in trainA (matching the name in trainB)
        target_path = os.path.join(output_dir, image_name)
        
        if os.path.exists(target_path):
            continue
            
        # Temporary directory for blender output (it saves to ./renders by default in the script)
        temp_renders_dir = "renders"
        if not os.path.exists(temp_renders_dir):
            os.makedirs(temp_renders_dir)
            
        # Run Blender
        # Note: we use small samples (16) and resolution (512) for speed, 
        # but you might want to increase them for better quality.
        cmd = [
            blender_path,
            blend_file,
            "--background",
            "--python", script_file,
            "--",
            "--fen", fen,
            "--resolution", "512",
            "--samples", "16",
            "--view", "white" # Or black, depending on the game. 
        ]
        
        try:
            # We use check=True to stop on errors. 
            # stdout=subprocess.DEVNULL to keep the console clean.
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # The script saves 1_overhead.png, 2_east.png, 3_west.png
            # We want 1_overhead.png
            overhead_src = os.path.join(temp_renders_dir, "1_overhead.png")
            if os.path.exists(overhead_src):
                os.rename(overhead_src, target_path)
            else:
                print(f"Warning: 1_overhead.png not found for FEN {fen}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error rendering FEN {fen}: {e}")

    print("Finished generating synthetic data.")

if __name__ == "__main__":
    generate_synthetic_data()

