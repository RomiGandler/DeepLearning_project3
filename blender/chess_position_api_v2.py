"""
Chess FEN Parser - Auto-detect Starting Positions
UPDATED: Fixed camera pointing to center (prevents grey images) + Custom Output Name
"""

import bpy
import math
from mathutils import Vector, Matrix
import sys
import argparse
import os

# ==========================
# CONFIG
# ==========================
REAL_BOARD_SIZE = 0.53
DESIRED_CAMERA_HEIGHT = 0.9   # original was 2
DESIRED_ANGLE_DEGREES = 0 # original was 25    
LENS = 50                     
RES = 800 # original was 1024
SAMPLES = 128
OUT_DIR = "//renders"

def get_board_info():
    """Get board dimensions"""
    plane = bpy.data.objects.get("Black & white")
    frame = bpy.data.objects.get("Outer frame")
    
    plane_pts = [plane.matrix_world @ Vector(v) for v in plane.bound_box]
    plane_min = Vector((min(p.x for p in plane_pts), min(p.y for p in plane_pts), min(p.z for p in plane_pts)))
    plane_max = Vector((max(p.x for p in plane_pts), max(p.y for p in plane_pts), max(p.z for p in plane_pts)))
    
    frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
    frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
    frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
    center = (frame_min + frame_max) / 2
    board_size = max(frame_max.x - frame_min.x, frame_max.y - frame_min.y)
    
    scale_factor = board_size / REAL_BOARD_SIZE

    # Calculate square size
    plane_size = max(plane_max.x - plane_min.x, plane_max.y - plane_min.y)
    square_size = plane_size / 8
    
    return {
        'square_size': square_size,
        'plane_min': plane_min,
        'plane_max': plane_max,
        'center': center,
        'scale_factor': scale_factor,
    }

def position_to_square(pos, board_info):
    """Convert 3D position to chess square (e.g., 'e2')"""
    square_size = board_info['square_size']
    plane_min = board_info['plane_min']
    plane_max = board_info['plane_max']
    
    file_idx = 7 - int((pos.x - plane_min.x) / square_size)
    file_idx = max(0, min(7, file_idx))
    file_letter = chr(ord('a') + file_idx)
    
    rank_idx = int((plane_max.y - pos.y) / square_size)
    rank_idx = max(0, min(7, rank_idx))
    rank_number = rank_idx + 1
    
    return f"{file_letter}{rank_number}"

def detect_starting_positions(board_info):
    """Detect which piece is on which square currently"""
    print("DETECTING STARTING POSITIONS...")
    pieces = {}
    
    for obj in bpy.data.objects:
        if obj.type != 'MESH': continue
        
        name = obj.name
        piece_type = None
        
        if name in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'A(texture)']: piece_type = 'P'
        elif name in ['B.001', 'C.001', 'D.001', 'E.001', 'F.001', 'G.001', 'H.001', 'A(textures)']: piece_type = 'p'
        elif 'rook' in name.lower(): piece_type = 'R' if 'white' in name.lower() else 'r'
        elif 'knight' in name.lower(): piece_type = 'N' if 'white' in name.lower() else 'n'
        elif 'bitshop' in name.lower() or 'bishop' in name.lower(): piece_type = 'B' if 'white' in name.lower() else 'b'
        elif 'queen' in name.lower(): piece_type = 'Q' if 'white' in name.lower() else 'q'
        elif 'king' in name.lower(): piece_type = 'K' if 'white' in name.lower() else 'k'
        
        if piece_type:
            square = position_to_square(obj.location, board_info)
            pieces[name] = {
                'square': square,
                'piece_type': piece_type,
                'start_pos': obj.location.copy()
            }
    
    print(f"✓ Detected {len(pieces)} pieces")
    return pieces

def parse_fen(fen):
    board_fen = fen.split()[0]
    ranks = board_fen.split('/')
    position = {}
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        board_rank = 8 - rank_idx
        for char in rank:
            if char.isdigit(): file_idx += int(char)
            else:
                file_letter = chr(ord('a') + file_idx)
                square = f"{file_letter}{board_rank}"
                position[square] = char
                file_idx += 1
    return position

def apply_fen(fen, starting_pieces, board_info):
    print(f"APPLYING FEN: {fen}")
    target_position = parse_fen(fen)
    square_size = board_info['square_size']
    
    pieces_used = set()
    
    for target_square, piece_type in target_position.items():
        candidates = []
        for piece_name, info in starting_pieces.items():
            if info['piece_type'] == piece_type and piece_name not in pieces_used:
                from_square = info['square']
                # מרחק פשוט (מנהטן)
                dist = abs((ord(target_square[0]) - ord(from_square[0]))) + abs((int(target_square[1]) - int(from_square[1])))
                candidates.append((dist, piece_name, from_square))
        
        if not candidates: continue
        
        candidates.sort()
        _, piece_name, from_square = candidates[0]
        
        obj = bpy.data.objects.get(piece_name)
        if obj:
            # Calculate displacement
            file_diff = (ord(target_square[0]) - ord('a')) - (ord(from_square[0]) - ord('a'))
            rank_diff = (int(target_square[1]) - 1) - (int(from_square[1]) - 1)
            
            obj.location.x -= file_diff * square_size
            obj.location.y -= rank_diff * square_size
            obj.hide_render = False
            obj.hide_viewport = False
            pieces_used.add(piece_name)
    
    for piece_name in starting_pieces.keys():
        if piece_name not in pieces_used:
            obj = bpy.data.objects.get(piece_name)
            if obj:
                obj.hide_render = True
                obj.hide_viewport = True

def render_all_views(board_info, view='black', output_name="render"):
    """
    Render ONLY the requested view (White or Black) with the correct angle.
    Forces camera to look at the center of the board.
    """
    print("\n" + "="*70)
    print(f"RENDERING ({view.upper()} VIEW)")
    print("="*70)
    
    center = board_info['center']
    scale_factor = board_info['scale_factor']

    # Calculate height and distance according to the CONFIG
    camera_height = DESIRED_CAMERA_HEIGHT * scale_factor
    angle_radians = math.radians(DESIRED_ANGLE_DEGREES)
    # Calculate the horizontal offset backwards to get the angle (if the angle is 0, the distance will be 0)
    horizontal_offset = camera_height * math.tan(angle_radians)

    # 1. Clear old cameras
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    # 2. Lighting (if none)
    if not any(o.type == "LIGHT" for o in bpy.data.objects):
        light_height = center.z + camera_height * 2
        bpy.ops.object.light_add(type="SUN", location=(center.x, center.y, light_height))
        bpy.context.active_object.data.energy = 4.0

    # 3. Render settings
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    scene.render.resolution_x = RES
    scene.render.resolution_y = RES
    scene.render.image_settings.file_format = 'PNG' # PNG for quality
    scene.cycles.use_denoising = True
    
    try:
        scene.cycles.device = 'GPU'
    except:
        pass
    
    # 4. Setup camera
    camera_z = center.z + camera_height
    
    if view == 'white':
        # White: Camera at positive Y, looking back
        cam_loc = (center.x, center.y + horizontal_offset, camera_z)
    else:
        # Black: Camera at negative Y, looking forward
        cam_loc = (center.x, center.y - horizontal_offset, camera_z)

    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object

    # --- Critical fix: Make the camera look at the center ---
    # This is what fixes the gray and out-of-focus images
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # If we're in white view, need to flip the camera 180 degrees on Z axis to not be reversed
    if view == 'white':
         cam.rotation_euler.z += math.radians(180)

    cam.data.lens = LENS
    bpy.context.scene.camera = cam

    # 5. Save the file with the correct name
    if not output_name.lower().endswith('.png'):
        output_name += ".png"
        
    full_path = os.path.join(OUT_DIR, output_name)
    scene.render.filepath = full_path
    
    print(f"  Rendering to: {full_path}...")
    bpy.ops.render.render(write_still=True)
    print(f"  ✓ Saved!")

def main():
    argv = sys.argv
    if "--" in argv: argv = argv[argv.index("--") + 1:]
    else: argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fen', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--view', type=str, default='black', choices=['white', 'black'])
    parser.add_argument('--output_name', type=str, default="render_output")

    args = parser.parse_args(argv)
    global RES, SAMPLES, OUT_DIR
    RES = args.resolution
    SAMPLES = args.samples
    OUT_DIR = "./renders"

    board_info = get_board_info()

    # Rotate the plane
    plane = bpy.data.objects.get("Black & white")
    if plane:
        frame = bpy.data.objects.get("Outer frame")
        # Recalculate center
        frame_pts = [frame.matrix_world @ Vector(v) for v in frame.bound_box]
        frame_min = Vector((min(p.x for p in frame_pts), min(p.y for p in frame_pts), min(p.z for p in frame_pts)))
        frame_max = Vector((max(p.x for p in frame_pts), max(p.y for p in frame_pts), max(p.z for p in frame_pts)))
        center = (frame_min + frame_max) / 2
        
        original_pos = plane.location.copy()
        offset = original_pos - center
        plane.rotation_euler.z = math.radians(90)
        rot_matrix = Matrix.Rotation(math.radians(90), 3, 'Z')
        rotated_offset = rot_matrix @ offset
        plane.location = center + rotated_offset
        
    starting_pieces = detect_starting_positions(board_info)
    apply_fen(args.fen, starting_pieces, board_info)
    render_all_views(board_info, view=args.view, output_name=args.output_name)

if __name__ == "__main__":
    main()