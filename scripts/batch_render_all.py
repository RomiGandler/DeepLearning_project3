import os
import subprocess

# Definitions   
BLENDER_PATH = "/Applications/Blender.app/Contents/MacOS/Blender"
BLEND_FILE = "blender/chess-set.blend"
SCRIPT_FILE = "blender/generate_cyclegan_data.py"
OUTPUT_BASE = "/Users/romigandler/Desktop/CS/Deep Learning/Project3/renders"
LIMIT = 3000  # Maximum images per game

# List of tasks to perform
# Each task is: (folder_name_to_create, path_to_csv_file)
tasks = [
    # --- Group 1: Games that arrived as PGN (after conversion) ---
    ("trainA_game9",  "pgn_data/game9/game9_converted.csv"),
    ("trainA_game10", "pgn_data/game10/game10_converted.csv"),
    ("trainA_game11", "pgn_data/game11/game11_converted.csv"),
    ("trainA_game12", "pgn_data/game12/game12_converted.csv"),
    ("trainA_game13", "pgn_data/game13/game13_converted.csv"),

    # --- Group 2: Games that arrived as CSV (frames) ---
    ("trainA_game4", "labeled_chess_data/game4_per_frame/game4.csv"),
    ("trainA_game5", "labeled_chess_data/game5_per_frame/game5.csv"),
    ("trainA_game6", "labeled_chess_data/game6_per_frame/game6.csv"),
    ("trainA_game7", "labeled_chess_data/game7_per_frame/game7.csv")
]

def run_renders():
    print(f"--- Starting Batch Render for {len(tasks)} games ---")
    print("Go grab a coffee, this will take a while... ☕")
    
    for folder_name, csv_path in tasks:
        full_output_dir = os.path.join(OUTPUT_BASE, folder_name)
        
        print(f"\n🎥 Rendering: {folder_name}")
        print(f"   Input: {csv_path}")

        # Check if the input exists
        if not os.path.exists(csv_path):
            print(f"❌ Error: CSV file not found: {csv_path}. Skipping.")
            continue

        # Build the command for Blender
        cmd = [
            BLENDER_PATH,
            BLEND_FILE,
            "--background",
            "--python", SCRIPT_FILE,
            "--",
            "--csv", csv_path,
            "--output_dir", full_output_dir,
            "--limit", str(LIMIT)
        ]

        # Run the command
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Finished {folder_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to render {folder_name}. Error: {e}")

    print("\n🎉 ALL RENDERS COMPLETE! 🎉")

if __name__ == "__main__":
    run_renders()