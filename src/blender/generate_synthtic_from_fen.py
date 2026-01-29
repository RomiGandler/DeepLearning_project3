"""
BBDN Data Generator (Calibrated 1:1) - FIX: Absolute Paths
File: blender/generate_synthtic_from_fen.py
==========================================================
"""

import bpy
import math
from mathutils import Vector
import sys
import argparse
import os
import csv

# ==========================
# 1:1 CALIBRATION CONSTANTS
# ==========================
BOARD_MIN_X = -21.8222
BOARD_MAX_X = -2.1417
BOARD_MIN_Y = -8.6489  
BOARD_MAX_Y = 11.0316
BOARD_Z = 0.7043

TOTAL_WIDTH = 19.6805
SQUARE_SIZE = 2.4601  

# Camera Settings
CAMERA_HEIGHT = 35.0 
LENS = 50
RES = 800
SAMPLES = 64

def get_square_center(file_idx, rank_idx):
    center_x = BOARD_MIN_X + (file_idx * SQUARE_SIZE) + (SQUARE_SIZE / 2)
    center_y = BOARD_MIN_Y + (rank_idx * SQUARE_SIZE) + (SQUARE_SIZE / 2)
    return center_x, center_y, BOARD_Z

def detect_pieces():
    pieces = {}
    for obj in bpy.data.objects:
        if obj.type != 'MESH': continue
        if "Black & white" in obj.name or "Outer frame" in obj.name: continue
        
        name = obj.name
        ptype = None
        
        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']: ptype = 'P'
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']: ptype = 'p'
        elif 'rook' in name.lower(): ptype = 'R' if 'white' in name.lower() else 'r'
        elif 'knight' in name.lower(): ptype = 'N' if 'white' in name.lower() else 'n'
        elif 'bitshop' in name.lower() or 'bishop' in name.lower(): ptype = 'B' if 'white' in name.lower() else 'b'
        elif 'queen' in name.lower(): ptype = 'Q' if 'white' in name.lower() else 'q'
        elif 'king' in name.lower(): ptype = 'K' if 'white' in name.lower() else 'k'
        
        if ptype:
            pieces[name] = {'type': ptype, 'obj': obj, 'base_z': obj.location.z}
            
    return pieces

def parse_fen(fen):
    board_fen = fen.split()[0]
    ranks = board_fen.split('/')
    position = {} 
    for r_idx, rank in enumerate(ranks):
        f_idx = 0
        real_rank = 7 - r_idx 
        for char in rank:
            if char.isdigit():
                f_idx += int(char)
            else:
                position[(f_idx, real_rank)] = char
                f_idx += 1
    return position

def apply_fen(fen, piece_map):
    target_pos = parse_fen(fen)
    
    available_pieces = {}
    for pname, pdata in piece_map.items():
        pdata['obj'].hide_render = True
        pdata['obj'].hide_viewport = True
        pdata['obj'].location.x = 100 
        
        ptype = pdata['type']
        if ptype not in available_pieces: available_pieces[ptype] = []
        available_pieces[ptype].append(pdata)

    for (f_idx, r_idx), ptype in target_pos.items():
        if ptype not in available_pieces or not available_pieces[ptype]:
            continue

        piece_data = available_pieces[ptype].pop()
        obj = piece_data['obj']

        tx, ty, tz = get_square_center(f_idx, r_idx)
        obj.location.x = tx
        obj.location.y = ty
        obj.location.z = piece_data['base_z']
        
        obj.hide_render = False
        obj.hide_viewport = False

def setup_camera():
    center_x = (BOARD_MIN_X + BOARD_MAX_X) / 2
    center_y = (BOARD_MIN_Y + BOARD_MAX_Y) / 2

    for o in bpy.data.objects:
        if o.type in ["CAMERA", "LIGHT"]: 
            bpy.data.objects.remove(o, do_unlink=True)

    bpy.ops.object.light_add(type="SUN", location=(center_x, center_y, CAMERA_HEIGHT))
    bpy.context.active_object.data.energy = 3.0

    bpy.ops.object.camera_add(location=(center_x, center_y, CAMERA_HEIGHT))
    cam = bpy.context.active_object
    cam.rotation_euler = (0, 0, 0) 
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
    parser.add_argument('--fen', type=str, help="FEN string to render")
    parser.add_argument('--csv', type=str, help="Optional: Path to CSV for batch generation")
    
    # Use 'output' folder by default instead of current dir to avoid permission issues
    parser.add_argument('--output_dir', type=str, default="output") 
    parser.add_argument('--output_name', type=str, default="test_result")
    
    args = parser.parse_args(argv)
    
    # --- CRITICAL FIX: Ensure Absolute Path & Create Directory ---
    abs_output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(abs_output_dir):
        try:
            os.makedirs(abs_output_dir)
            print(f"Created output directory: {abs_output_dir}")
        except Exception as e:
            print(f"Error creating directory {abs_output_dir}: {e}")
            return

    piece_map = detect_pieces()
    setup_camera()
    
    if args.fen:
        # --- TEST MODE ---
        print(f"Generating Single FEN: {args.fen}")
        apply_fen(args.fen, piece_map)
        
        fname = args.output_name
        if not fname.lower().endswith('.png'): fname += ".png"
        
        # Combine absolute dir with filename
        fpath = os.path.join(abs_output_dir, fname)
        
        bpy.context.scene.render.filepath = fpath
        bpy.ops.render.render(write_still=True)
        print(f"SUCCESS: Saved to {fpath}")

    elif args.csv:
        # --- BATCH MODE ---
        print(f"Processing CSV batch...")
        unique_fens = set()
        count = 0
        with open(args.csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fen = row['fen']
                if fen in unique_fens: continue
                unique_fens.add(fen)
                apply_fen(fen, piece_map)
                
                fpath = os.path.join(abs_output_dir, f"synthetic_{count:04d}.png")
                bpy.context.scene.render.filepath = fpath
                bpy.ops.render.render(write_still=True)
                count += 1
    else:
        print("ERROR: Provide --fen or --csv")

if __name__ == "__main__":
    main()