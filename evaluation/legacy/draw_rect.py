import cv2
import sys
import os

def draw_grid(image_path):
    """
    Loads an image, conceptually divides it into an 8x8 grid (64 squares),
    draws the grid boundaries on the image, and saves it.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    height, width = img.shape[:2]
    
    # Grid dimensions (8x8)
    rows = 8
    cols = 8
    
    dy = height / rows
    dx = width / cols
    
    # Draw vertical lines
    for x in range(cols + 1):
        # Calculate x coordinate
        # Using int() for pixel coordinates
        px = int(x * dx)
        start_point = (px, 0)
        end_point = (px, height)
        color = (0, 255, 0) # Green
        thickness = 2
        cv2.line(img, start_point, end_point, color, thickness)
        
    # Draw horizontal lines
    for y in range(rows + 1):
        # Calculate y coordinate
        py = int(y * dy)
        start_point = (0, py)
        end_point = (width, py)
        color = (0, 255, 0) # Green
        thickness = 2
        cv2.line(img, start_point, end_point, color, thickness)
        
    # Save the result
    # Splitting extension to handle it correctly
    base_name = os.path.splitext(image_path)[0]
    output_path = f"{base_name}_with_grid.jpg"
    
    cv2.imwrite(output_path, img)
    print(f"Saved grid image to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to a known image if available, for convenience
        default_img = "game2_frame_000896.png"
        if os.path.exists(default_img):
            image_path = default_img
        elif os.path.exists(os.path.join("evaluation", default_img)):
            image_path = os.path.join("evaluation", default_img)
        else:
            print("Usage: python draw_rect.py <image_path>")
            sys.exit(1)
            
    draw_grid(image_path)
