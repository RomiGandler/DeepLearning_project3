# ControlNet Guide for Chess Images

## What is ControlNet?

**ControlNet** allows you to guide Stable Diffusion image generation using structural information (like edges, depth, poses, etc.)

For chess images:
- **Input**: Your synthetic chess renders
- **Control**: Canny edges (extracted automatically from your synthetic images)
- **Output**: Photorealistic chess images that match the structure/layout

### vs CycleGAN
- **CycleGAN**: Learns style transfer from unpaired data (what you trained)
- **ControlNet**: Uses pre-trained Stable Diffusion + control signals (no training needed!)

---

## Quick Start

### 1. Run the Script

```bash
cd /home/nessm/DeepLearning_project3/cycleGAN
python controlnet_chess.py
```

This will:
- Process first 5 synthetic chess images from `datasets/chess_data/trainA/`
- Extract Canny edges as control signals
- Generate realistic images using ControlNet
- Save results to `controlnet_chess_results/`

### 2. View Results

Check the output directory:
```bash
ls controlnet_chess_results/
```

For each input image, you'll get:
- `*_canny.png` - Edge map (control signal)
- `*_realistic.png` - Generated realistic image

---

## How It Works

```
Synthetic Chess Image
        ↓
[Extract Canny Edges] ← Control signal
        ↓
[ControlNet + Stable Diffusion]
   + Text Prompt: "realistic chess board..."
        ↓
Realistic Chess Image
```

**Key advantage**: Uses the powerful pre-trained Stable Diffusion model (billions of images) + your specific chess layout!

---

## Customization

Edit `controlnet_chess.py` to adjust:

### 1. Number of Images
```python
MAX_IMAGES = 5  # Change to process more images
```

### 2. Text Prompt (Important!)
```python
prompt="a realistic wooden chess board with wooden chess pieces, professional photography, high quality, detailed, natural lighting"
```

Try different prompts:
- `"marble chess set on a marble table, luxury, professional photography"`
- `"glass chess pieces on a modern chess board, studio lighting"`
- `"outdoor chess board in a park, natural daylight"`

### 3. Quality Settings
```python
num_inference_steps=20  # Higher = better quality but slower (try 30-50)
seed=42                 # Change for different results
```

### 4. Edge Detection Sensitivity
```python
low_threshold=100   # Lower = more edges detected
high_threshold=200  # Adjust for different edge sensitivity
```

---

## Comparison: 3 Methods for Your Chess Project

| Method | Training Required | Data Requirement | Quality | Speed | Best For |
|--------|------------------|------------------|---------|-------|----------|
| **CycleGAN** | ✅ Yes (~3 hrs) | Unpaired images | Good | Medium | Style transfer without correspondence |
| **pix2pix** | ✅ Yes (~3 hrs) | Paired images | Better | Medium | Direct translation with exact pairs |
| **ControlNet** | ❌ No | Any single image | Excellent | Slow | High-quality results with no training |

---

## Expected Performance

### GPU (GTX 1080 Ti):
- **Per image**: ~10-15 seconds (20 steps)
- **5 images**: ~1 minute
- **100 images**: ~15-20 minutes

### CPU:
- ⚠️ **Very slow** (~5-10 minutes per image)
- Not recommended for batch processing

---

## Tips for Best Results

### 1. Prompt Engineering
The text prompt is **crucial** for quality:

**Good prompts**:
- ✅ "realistic wooden chess board, professional photography, detailed"
- ✅ "marble chess set, studio lighting, high resolution, 8k"

**Bad prompts**:
- ❌ "chess" (too vague)
- ❌ "3d render chess" (will generate synthetic-looking images)

### 2. Negative Prompts
Specify what you DON'T want:
```python
negative_prompt="cartoon, synthetic, 3d render, cgi, blurry, low quality"
```

### 3. Inference Steps
- **20 steps**: Fast, decent quality
- **30-40 steps**: Balanced
- **50+ steps**: Best quality, slower

---

## Advantages Over CycleGAN

**ControlNet Advantages**:
1. ✅ **No training required** - Use immediately
2. ✅ **Better quality** - Leverages pre-trained Stable Diffusion
3. ✅ **Flexible** - Change style via text prompts
4. ✅ **Precise control** - Edges ensure exact layout preservation

**CycleGAN Advantages**:
1. ✅ **Faster inference** - Once trained
2. ✅ **Consistent style** - Learned from your specific dataset
3. ✅ **No prompt engineering** - Automatic style transfer

---

## Batch Processing All Images

To process all synthetic images:

```python
# Edit controlnet_chess.py, change:
MAX_IMAGES = 1000  # Or however many you have

# Then run:
python controlnet_chess.py
```

---

## Memory Issues?

If you run out of GPU memory:

### Option 1: Reduce Image Size
```python
# In controlnet_chess.py, resize images before processing
image = Image.open(image_path).convert('RGB')
image = image.resize((512, 512))  # Smaller size
```

### Option 2: Use CPU Offloading (already enabled)
The script uses `pipe.enable_model_cpu_offload()` to save GPU memory

### Option 3: Process Fewer Images at Once
Change `MAX_IMAGES` to a smaller number

---

## Troubleshooting

### Error: "CUDA out of memory"
- Reduce inference steps
- Process fewer images
- Close other programs using GPU

### Error: "Model download failed"
- Check internet connection
- Models are ~5GB each (ControlNet + Stable Diffusion)
- First run downloads models automatically

### Generated images don't look good
- Improve text prompt
- Increase inference steps (30-50)
- Adjust Canny thresholds
- Try different seeds

---

## Next Steps

1. **Run the script**: `python controlnet_chess.py`
2. **Compare results**: 
   - ControlNet vs CycleGAN (epoch 37)
   - See which method works better for your project
3. **Optimize prompts**: Experiment with different text prompts
4. **Scale up**: Process all 1000 synthetic images if results are good

---

## File Locations

**Script**: `/home/nessm/DeepLearning_project3/cycleGAN/controlnet_chess.py`

**Input**: `./datasets/chess_data/trainA/` (synthetic images)

**Output**: `./controlnet_chess_results/`

**Models** (auto-downloaded first run):
- `~/.cache/huggingface/` (~5GB)

---

## Summary

**ControlNet is great for**:
- Quick experimentation without training
- High-quality results
- Flexible style control via text

**Use CycleGAN when**:
- You want consistent style from your specific dataset
- You need fast inference for many images
- You prefer automatic style learning

**Try both and compare!** 🎯♟️

