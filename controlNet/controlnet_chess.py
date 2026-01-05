"""
ControlNet Canny Edge for Chess Image Enhancement
Uses ControlNet to generate realistic chess images from synthetic renders
"""

import torch
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Check if we have the required packages
try:
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )
    print("✅ Diffusers imported successfully")
except ImportError as e:
    print(f"❌ Error importing diffusers: {e}")
    print("Please ensure diffusers is installed: pip install diffusers")
    exit(1)

def extract_canny_edges(image_path, low_threshold=100, high_threshold=200):
    """
    Extract Canny edges from an image
    
    Args:
        image_path: Path to input image
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Upper threshold for Canny edge detection
    
    Returns:
        PIL Image with Canny edges
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    return Image.fromarray(edges)

def setup_controlnet_pipeline(device="cuda"):
    """
    Setup ControlNet pipeline with Stable Diffusion
    
    Args:
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Configured pipeline
    """
    print("Loading ControlNet model...")
    checkpoint = "lllyasviel/control_v11p_sd15_canny"
    
    # Use float16 for GPU, float32 for CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        controlnet = ControlNetModel.from_pretrained(
            checkpoint, 
            torch_dtype=dtype
        )
        print("✅ ControlNet loaded")
        
        print("Loading Stable Diffusion pipeline...")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=dtype
        )
        print("✅ Stable Diffusion loaded")
        
        # Use efficient scheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Enable CPU offloading to save GPU memory
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

def generate_realistic_chess_image(
    pipe,
    input_image_path,
    output_dir="controlnet_results",
    prompt="a realistic wooden chess board with chess pieces, professional photography, high quality, detailed",
    negative_prompt="cartoon, synthetic, 3d render, blurry, low quality",
    num_inference_steps=20,
    seed=42,
    low_threshold=100,
    high_threshold=200
):
    """
    Generate realistic chess image from synthetic render using ControlNet
    
    Args:
        pipe: ControlNet pipeline
        input_image_path: Path to synthetic chess image
        output_dir: Directory to save results
        prompt: Text prompt for generation
        negative_prompt: Negative prompt
        num_inference_steps: Number of inference steps
        seed: Random seed for reproducibility
        low_threshold: Canny lower threshold
        high_threshold: Canny upper threshold
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get base filename
    base_name = Path(input_image_path).stem
    
    print(f"\nProcessing: {input_image_path}")
    print(f"Prompt: {prompt}")
    
    # Extract Canny edges
    print("Extracting Canny edges...")
    control_image = extract_canny_edges(
        input_image_path, 
        low_threshold, 
        high_threshold
    )
    
    # Save control image
    control_save_path = output_path / f"{base_name}_canny.png"
    control_image.save(control_save_path)
    print(f"✅ Canny edges saved: {control_save_path}")
    
    # Generate image
    print(f"Generating realistic image (steps={num_inference_steps})...")
    generator = torch.manual_seed(seed)
    
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        image=control_image
    ).images[0]
    
    # Save generated image
    output_save_path = output_path / f"{base_name}_realistic.png"
    result.save(output_save_path)
    print(f"✅ Generated image saved: {output_save_path}")
    
    return result, control_image

def process_directory(
    pipe,
    input_dir,
    output_dir="controlnet_results",
    max_images=10,
    **kwargs
):
    """
    Process multiple images from a directory
    
    Args:
        pipe: ControlNet pipeline
        input_dir: Directory containing synthetic chess images
        output_dir: Directory to save results
        max_images: Maximum number of images to process
        **kwargs: Additional arguments for generate_realistic_chess_image
    """
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    
    print(f"\nFound {len(image_files)} images in {input_dir}")
    print(f"Processing first {min(max_images, len(image_files))} images...\n")
    
    for i, image_file in enumerate(image_files[:max_images]):
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{min(max_images, len(image_files))}")
        print(f"{'='*60}")
        
        try:
            generate_realistic_chess_image(
                pipe,
                str(image_file),
                output_dir=output_dir,
                **kwargs
            )
        except Exception as e:
            print(f"❌ Error processing {image_file}: {e}")
            continue

def main():
    """Main function"""
    print("="*60)
    print("ControlNet Canny - Chess Image Enhancement")
    print("="*60)
    print()
    
    # Configuration
    INPUT_DIR = "/home/nessm/DeepLearning_project3/chess_data/trainA"
    OUTPUT_DIR = "/home/nessm/DeepLearning_project3/controlNet/controlnet_chess_results"
    MAX_IMAGES = 1  # Force to 1 image for CPU test
    
    # Force CPU usage for stability
    device = "cpu"
    print(f"Using device: {device}")
    print("⚠️  Running on CPU - slow but stable! Expected time: 5-8 minutes.")
    
    print()
    
    # Setup pipeline
    pipe = setup_controlnet_pipeline(device)
    
    if pipe is None:
        print("❌ Failed to load pipeline")
        return
    
    print()
    print("="*60)
    print("Starting image processing...")
    print("="*60)
    
    # Process images
    process_directory(
        pipe,
        INPUT_DIR,
        OUTPUT_DIR,
        max_images=MAX_IMAGES,
        prompt="a realistic wooden chess board with wooden chess pieces, professional photography, high quality, detailed, natural lighting",
        negative_prompt="cartoon, synthetic, 3d render, cgi, blurry, low quality, artificial",
        num_inference_steps=20,
        seed=42
    )
    
    print()
    print("="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nFor each input image, you'll find:")
    print("  - *_canny.png     : Canny edge map (control signal)")
    print("  - *_realistic.png : Generated realistic image")
    print()

if __name__ == "__main__":
    main()

