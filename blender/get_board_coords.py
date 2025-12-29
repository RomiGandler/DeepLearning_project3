import bpy
from mathutils import Vector

def print_debug_info():
    print("\n" + "="*40)
    print("      BLENDER BOARD DIAGNOSTIC      ")
    print("="*40)
    
    # find the board object
    board = bpy.data.objects.get("Black & white")
    if not board:
        print("ERROR: Could not find object 'Black & white'")
        print("Available objects:")
        for o in bpy.data.objects:
            print(f"- {o.name}")
        return

    # Calculate the actual bounds of the board in world space
    bbox_corners = [board.matrix_world @ Vector(corner) for corner in board.bound_box]
    
    min_x = min(v.x for v in bbox_corners)
    max_x = max(v.x for v in bbox_corners)
    min_y = min(v.y for v in bbox_corners)
    max_y = max(v.y for v in bbox_corners)
    max_z = max(v.z for v in bbox_corners)
    
    print(f"X Range (Width): {min_x:.4f} to {max_x:.4f}")
    print(f"Y Range (Height): {min_y:.4f} to {max_y:.4f}")
    print(f"Z Height (Floor): {max_z:.4f}")
    
    width = max_x - min_x
    print(f"Total Width: {width:.4f}")
    print(f"Calculated Square Size: {width/8:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    print_debug_info()