import os
import subprocess
import sys

# List of game IDs to convert
games_to_convert = [9, 10, 11, 12, 13]

def convert_all():
    print("--- Starting Batch Conversion (PGN -> CSV) ---")

    # Check where the converter script is located
    # Try to find it in the scripts folder or the root folder
    if os.path.exists("scripts/pgn_to_csv.py"):
        converter_script = "scripts/pgn_to_csv.py"
    elif os.path.exists("pgn_to_csv.py"):
        converter_script = "pgn_to_csv.py"
    else:
        print("❌ Error: Could not find 'pgn_to_csv.py' in root or scripts folder.")
        return

    print(f"DEBUG: Found converter script at: {converter_script}")

    for game_id in games_to_convert:
        # Path definitions
        pgn_file = f"pgn_data/game{game_id}/game{game_id}.pgn"
        output_csv = f"pgn_data/game{game_id}/game{game_id}_converted.csv"

        # Check if the PGN file exists
        if not os.path.exists(pgn_file):
            print(f"⚠️ Warning: {pgn_file} not found. Skipping.")
            continue
            
        print(f"Converting Game {game_id}...")

        # Build the command
        cmd = [sys.executable, converter_script, pgn_file, output_csv]

        # Run with error checking
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Game {game_id} converted successfully.")
        else:
            print(f"❌ Failed to convert Game {game_id}.")
            print(f"   Error: {result.stderr}")

if __name__ == "__main__":
    convert_all()