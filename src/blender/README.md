# Blender Scripts

Scripts for generating synthetic chessboard images using Blender.

## Overview

These scripts use Blender to render 3D chessboard images from FEN notation. The generated synthetic images serve as input conditions for the BBDM model.

## Prerequisites

1. **Blender 5.0+** installed (or adjust path in config)
2. **Chess set blend file** (`blender/chess-set.blend`)

## Scripts

### 1. generate_synthtic_from_fen.py

Renders synthetic chessboard images from FEN positions.

#### Single Image Generation

```bash
blender -b blender/chess-set.blend \
    -P src/blender/generate_synthtic_from_fen.py \
    -- \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --output_dir ./results \
    --output_name synthetic
```

#### Batch Generation from CSV

```bash
blender -b blender/chess-set.blend \
    -P src/blender/generate_synthtic_from_fen.py \
    -- \
    --csv path/to/positions.csv \
    --output_dir ./output
```

CSV format:
```csv
fen,other_columns...
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,...
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--fen` | FEN string to render | None |
| `--csv` | Path to CSV for batch generation | None |
| `--output_dir` | Output directory | `output` |
| `--output_name` | Output filename (without extension) | `test_result` |

#### Render Settings

Configured in the script:
- **Resolution**: 800×800
- **Samples**: 64 (Cycles)
- **Camera**: Top-down orthographic view
- **Lens**: 50mm

### 2. crop_board.py

Crops rendered images to the chessboard area only (removes frame/background).

#### Preview Mode (Calibration)

```bash
python -m src.blender.crop_board --image results/synthetic.png --preview
```

This draws a red rectangle showing the crop area. Adjust `CROP_*` constants in the script if needed.

#### Single Image Crop

```bash
# Overwrite original
python -m src.blender.crop_board --image results/synthetic.png

# Save to different location
python -m src.blender.crop_board --image results/synthetic.png --output_dir cropped/
```

#### Batch Crop

```bash
python -m src.blender.crop_board --dir path/to/images/ --output_dir cropped/
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `--image` | Path to single image |
| `--dir` | Path to directory of images |
| `--output_dir` | Output directory (default: overwrite original) |
| `--preview` | Preview mode - draw crop rectangle only |

#### Crop Calibration

Default crop boundaries (adjust in script if needed):
```python
CROP_Y_START = 65   # Top
CROP_Y_END   = 735  # Bottom
CROP_X_START = 75   # Left
CROP_X_END   = 725  # Right
```

## Integration with submission.py

The submission script uses these functions internally:

```python
from src.blender.crop_board import process_single_image

# Generate synthetic (via subprocess to Blender)
subprocess.run([
    blender_exec, "-b", blend_file,
    "-P", "src/blender/generate_synthtic_from_fen.py",
    "--", "--fen", fen, "--output_dir", results_dir
])

# Crop to board area
process_single_image(path_synthetic, output_dir=None, preview_mode=False)
```

## Programmatic Usage

```python
from src.blender.crop_board import process_single_image

# Crop single image (overwrites original)
process_single_image("synthetic.png")

# Crop to new location
process_single_image("synthetic.png", output_dir="cropped/")

# Preview mode
process_single_image("synthetic.png", preview_mode=True)
```

## Troubleshooting

### "Cannot read blend file"
- Ensure `blender/chess-set.blend` exists
- Check path in `submission_config.yaml`

### Pieces not positioned correctly
- Verify FEN string format is valid
- Check calibration constants in `generate_synthtic_from_fen.py`

### Crop area wrong
1. Run with `--preview` flag
2. Adjust `CROP_*` constants in `crop_board.py`
3. Re-run preview until rectangle aligns with board
