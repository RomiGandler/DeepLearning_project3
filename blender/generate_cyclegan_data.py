"""
CycleGAN Data Generator (Calibrated)
====================================
Generates synthetic chess images using EXACT coordinates from diagnostic run.
"""

import bpy
import math
from mathutils import Vector
import sys
import argparse
import os
import csv
import random

# ==========================
# CALIBRATED CONSTANTS
# Constants derived from the diagnostic run
BOARD_MIN_X = -21.8222
BOARD_MAX_X = -2.1417
BOARD_MIN_Y = -8.6489  # (Calculated from Max Y - Width)
BOARD_MAX_Y = 11.0316
BOARD_Z = 0.7043

TOTAL_WIDTH = 19.6805
SQUARE_SIZE = 2.4601  # Total Width / 8

# Camera
# The board is huge (width 20), so the camera needs to be high
CAMERA_HEIGHT = 35.0 
LENS = 50
RES = 800
SAMPLES = 64

def get_square_center(file_idx, rank_idx):
    """
    file_idx: 0 (a) to 7 (h)
    rank_idx: 0 (1) to 7 (8)
    """
    # X Axis: Maps a..h
    # path generally goes from negative (a) to positive (h)
    center_x = BOARD_MIN_X + (file_idx * SQUARE_SIZE) + (SQUARE_SIZE / 2)
    
    # Y Axis: Maps 1..8
    # In Blender, positive Y is "up" (black), negative Y is "down" (white)
    center_y = BOARD_MIN_Y + (rank_idx * SQUARE_SIZE) + (SQUARE_SIZE / 2)
    
    return center_x, center_y, BOARD_Z

def detect_pieces():
    """Maps piece names to their types"""
    pieces = {}
    for obj in bpy.data.objects:
        if obj.type != 'MESH': continue

        # Skip the board and frame
        if "Black & white" in obj.name or "Outer frame" in obj.name: continue
        
        name = obj.name
        ptype = None
        
        # Mapping logic
        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']: ptype = 'P' # White Pawn
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']: ptype = 'p' # Black Pawn
        elif 'rook' in name.lower(): ptype = 'R' if 'white' in name.lower() else 'r'
        elif 'knight' in name.lower(): ptype = 'N' if 'white' in name.lower() else 'n'
        elif 'bitshop' in name.lower() or 'bishop' in name.lower(): ptype = 'B' if 'white' in name.lower() else 'b'
        elif 'queen' in name.lower(): ptype = 'Q' if 'white' in name.lower() else 'q'
        elif 'king' in name.lower(): ptype = 'K' if 'white' in name.lower() else 'k'
        
        if ptype:
            # Save the object and its original height (in case it differs between pieces)
            pieces[name] = {'type': ptype, 'obj': obj, 'base_z': obj.location.z}
            
    return pieces

def parse_fen(fen):
    board_fen = fen.split()[0]
    ranks = board_fen.split('/')
    position = {} 
    for r_idx, rank in enumerate(ranks):
        f_idx = 0
        real_rank = 7 - r_idx # 0 to 7 (where 7 is rank 8)
        for char in rank:
            if char.isdigit():
                f_idx += int(char)
            else:
                position[(f_idx, real_rank)] = char
                f_idx += 1
    return position

def apply_fen(fen, piece_map):
    target_pos = parse_fen(fen)
    
    # 1. Reset all pieces (Hide & Move away)
    # We're categorizing the pieces by type for retrieval
    available_pieces = {}
    
    for pname, pdata in piece_map.items():
        pdata['obj'].hide_render = True
        pdata['obj'].hide_viewport = True
        pdata['obj'].location.x = 100 # Move far away
        
        ptype = pdata['type']
        if ptype not in available_pieces: available_pieces[ptype] = []
        available_pieces[ptype].append(pdata)

    # 2. Place pieces
    for (f_idx, r_idx), ptype in target_pos.items():
        if ptype not in available_pieces or not available_pieces[ptype]:
            # If we run out of a certain piece type (happens in promotion sometimes)
            continue

        # Take a piece from the pool
        piece_data = available_pieces[ptype].pop()
        obj = piece_data['obj']

        # Calculate position
        tx, ty, tz = get_square_center(f_idx, r_idx)
        
        obj.location.x = tx
        obj.location.y = ty
        obj.location.z = piece_data['base_z'] # Keep the original height of the piece
        
        obj.hide_render = False
        obj.hide_viewport = False

def setup_camera():
    # Calculate the center of the board for camera positioning
    center_x = (BOARD_MIN_X + BOARD_MAX_X) / 2
    center_y = (BOARD_MIN_Y + BOARD_MAX_Y) / 2

    # Clear old cameras
    for o in bpy.data.objects:
        if o.type in ["CAMERA", "LIGHT"]: 
            bpy.data.objects.remove(o, do_unlink=True)

    # Lighting
    bpy.ops.object.light_add(type="SUN", location=(center_x, center_y, CAMERA_HEIGHT))
    bpy.context.active_object.data.energy = 3.0 # Strong lighting for large board

    # Camera
    bpy.ops.object.camera_add(location=(center_x, center_y, CAMERA_HEIGHT))
    cam = bpy.context.active_object
    cam.rotation_euler = (0, 0, 0) # Top-down view
    cam.data.lens = LENS
    bpy.context.scene.camera = cam
    
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    try: scene.cycles.device = 'GPU'
    except: pass

def main():
    argv = sys.argv
    if "--" in argv: argv = argv[argv.index("--") + 1:]
    else: argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="//output")
    parser.add_argument('--limit', type=int, default=2000)
    
    args = parser.parse_args(argv)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"DEBUG: Using Calibrated Board: X[{BOARD_MIN_X} to {BOARD_MAX_X}]")

    piece_map = detect_pieces()
    setup_camera()
    
    print(f"Starting generation from {args.csv}...")
    unique_fens = set()
    count = 0
    
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if count >= args.limit: break
            fen = row['fen']
            if fen in unique_fens: continue
            unique_fens.add(fen)
            
            apply_fen(fen, piece_map)
            
            fname = f"synthetic_{count:04d}.png"
            fpath = os.path.join(args.output_dir, fname)
            bpy.context.scene.render.filepath = fpath
            bpy.ops.render.render(write_still=True)
            
            print(f"Saved {fname}")
            count += 1

if __name__ == "__main__":
    main()