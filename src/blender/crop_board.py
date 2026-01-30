import cv2
import os
import argparse
import glob

# ==========================================
# Cropping calibration parameters
# ==========================================
# Adjust these values until the red rectangle (in preview mode)
# perfectly aligns with the chessboard area only (without frame or background).

CROP_Y_START = 75   # Top boundary
CROP_Y_END   = 725  # Bottom boundary

CROP_X_START = 75   # Left boundary
CROP_X_END   = 725  # Right boundary
# ==========================================

def process_single_image(image_path, output_dir=None, preview_mode=False):
    """
    Process a single image:
    - In preview mode: draw a red rectangle for calibration.
    - In action mode: crop the image according to the defined boundaries.
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image {image_path}")
        return

    # === Preview mode: draw red rectangle only ===
    if preview_mode:
        cv2.rectangle(
            img,
            (CROP_X_START, CROP_Y_START),
            (CROP_X_END, CROP_Y_END),
            (0, 0, 255),  # Red color in BGR
            2
        )

        preview_name = "preview_calibration.png"
        cv2.imwrite(preview_name, img)
        print(f"Preview mode: saved calibration image to '{preview_name}'")
        print("Verify the red box alignment and adjust crop values if needed.")
        return

    # === Action mode: perform actual cropping ===
    cropped_img = img[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

    # Determine save path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, filename)
    else:
        # Overwrite original image
        save_path = image_path

    cv2.imwrite(save_path, cropped_img)
    print(
        f"Cropped and saved: {save_path} "
        f"(New size: {cropped_img.shape[1]}x{cropped_img.shape[0]})"
    )

def main():
    parser = argparse.ArgumentParser(description="Crop chessboard area from images.")

    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--dir", type=str, help="Path to a directory of images")
    parser.add_argument("--output_dir", type=str, help="Optional output directory")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview crop area by drawing a red rectangle (no cropping)"
    )

    args = parser.parse_args()

    if args.image:
        process_single_image(args.image, args.output_dir, args.preview)

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: Directory {args.dir} not found")
            return

        images = glob.glob(os.path.join(args.dir, "*.png"))
        print(f"Found {len(images)} images. Processing...")

        for img_path in images:
            process_single_image(img_path, args.output_dir, args.preview)
    else:
        print("Usage: python crop_board.py --image <path> [--preview]")

if __name__ == "__main__":
    main()
