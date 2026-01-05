import os
from PIL import Image
from tqdm import tqdm

def resize_images(directory, size=(256, 256)):
    print(f"Resizing images in {directory} to {size}...")
    files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in tqdm(files):
        filepath = os.path.join(directory, filename)
        try:
            with Image.open(filepath) as img:
                # Convert to RGB if necessary (PNGs might be RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_resized = img.resize(size, Image.LANCZOS)
                
                # Overwrite with resized version (save as JPG for consistency and space)
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                new_filepath = os.path.join(directory, new_filename)
                
                img_resized.save(new_filepath, "JPEG", quality=90)
                
                # If we changed extension from .png to .jpg, remove original
                if filename.lower().endswith('.png'):
                    os.remove(filepath)
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    resize_images("dataset/trainA")
    resize_images("dataset/trainB")
    print("Resizing complete.")

