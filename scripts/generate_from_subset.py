import os
import json
import subprocess
from tqdm import tqdm
import sys

def generate_synthetic_data():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_file = os.path.join(base_dir, "chessred2k", "images", "subset_fen_pairs.json")
    output_dir = os.path.join(base_dir, "synthetic_dataset")
    
    # Blender path - Detected from previous step
    blender_path = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
    
    blend_file = os.path.join(base_dir, "blender", "chess-set.blend")
    script_file = os.path.join(base_dir, "blender", "chess_position_api_v2.py")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found.")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
    print(f"Starting generation of {len(metadata)} synthetic images...")
    
    # Process only a few for testing if needed, or all
    # For now, let's try to process all, but user can stop it
    
    for entry in tqdm(metadata):
        image_relative_path = entry['image_path'] # e.g. "images/0/G000_IMG001.jpg"
        # We might want to flatten the structure or keep it. Let's keep it.
        # But wait, "images/0/..." implies we might overwrite if we are not careful?
        # The JSON has paths relative to the chessred2k root? 
        # Actually in extract_fen.py we saw: "images/0/G000_IMG001.jpg"
        
        # Let's map "images/0/XXX.jpg" to "synthetic_dataset/images/0/XXX.png" (Blender renders PNG usually)
        # Note: Blender script outputs PNG. The input name is JPG. We should change extension.
        
        target_rel_path = os.path.splitext(image_relative_path)[0] + ".png"
        target_path = os.path.join(output_dir, target_rel_path)
        
        # Ensure target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        if os.path.exists(target_path):
            continue
            
        fen = entry['fen']
        
        # Temporary directory for blender output
        temp_renders_dir = os.path.join(base_dir, "renders")
        if not os.path.exists(temp_renders_dir):
            os.makedirs(temp_renders_dir)
            
        # Run Blender
        cmd = [
            blender_path,
            blend_file,
            "--background",
            "--python", script_file,
            "--",
            "--fen", fen,
            "--resolution", "512",
            "--samples", "16", # Low samples for speed
            "--view", "white" 
        ]
        
        try:
            # Run Blender
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # The script saves 1_overhead.png in the temp directory (or current dir?)
            # Usually chess_position_api_v2.py saves to where it is run or specified output.
            # Let's check where it saves. 
            # Looking at chess_position_api_v2.py snippet from before: 
            # output_name="render" -> "render_1_overhead.png"? 
            # Wait, the previous script looked for "renders/1_overhead.png". 
            # Let's assume the script defaults to "renders" dir in CWD.
            
            # We need to run subprocess with cwd=base_dir to be safe
            
            overhead_src = os.path.join(temp_renders_dir, "1_overhead.png")
            
            # If not found, check just "1_overhead.png" in CWD
            if not os.path.exists(overhead_src):
                overhead_src = os.path.join(base_dir, "1_overhead.png")

            if os.path.exists(overhead_src):
                os.rename(overhead_src, target_path)
            else:
                # print(f"Warning: Output not found for FEN {fen}")
                pass
                
        except subprocess.CalledProcessError as e:
            print(f"Error rendering FEN {fen}: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Finished generating synthetic data.")

if __name__ == "__main__":
    generate_synthetic_data()

