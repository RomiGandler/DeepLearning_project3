# Blender Scripts

Scripts for generating synthetic chessboard images using Blender.

## Overview

These scripts use Blender to render 3D chessboard images from FEN notation. The generated synthetic images serve as input conditions for the BBDM model.

## Prerequisites

1. **Blender 5.0+** installed (or adjust path in config)
2. **Chess set blend file** (`blender/chess-set.blend`)

## Scripts

### 1. generate_synthetic_from_fen.py

Renders synthetic chessboard images from FEN positions.

#### Single Image Generation

```bash
blender -b blender/chess-set.blend \
    -P etl/blender/generate_synthetic_from_fen.py \
    -- \
    --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
    --output_dir ./results \
    --output_name synthetic
```

#### Batch Generation from CSV

```bash
blender -b blender/chess-set.blend \
    -P etl/blender/generate_synthetic_from_fen.py \
    -- \
    --csv path/to/positions.csv \
    --output_dir ./output \
    --fen_column FEN \
    --img_name_column IMG_NAME
```

CSV format:
```csv
FEN,IMG_NAME
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR,game1_frame_000001.png
...
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--fen` | FEN string to render | None |
| `--csv` | Path to CSV for batch generation | None |
| `--output_dir` | Output directory | `output` |
| `--output_name` | Output filename (single mode) | `test_result` |
| `--fen_column` | Column name for FEN strings | `FEN` |
| `--img_name_column` | Column name for output filenames | `IMG_NAME` |

#### Render Settings

Configured in the script:
- **Resolution**: 800x800
- **Samples**: 64 (Cycles)
- **Camera**: Top-down orthographic view
- **Lens**: 50mm

### 2. crop_board.py

Crops rendered images to the chessboard area only (removes frame/background).

#### Preview Mode (Calibration)

```bash
python -m etl.blender.crop_board --image results/synthetic.png --preview
```

This draws a red rectangle showing the crop area. Adjust `CROP_*` constants in the script if needed.

#### Single Image Crop

```bash
# Overwrite original
python -m etl.blender.crop_board --image results/synthetic.png

# Save to different location
python -m etl.blender.crop_board --image results/synthetic.png --output_dir cropped/
```

#### Batch Crop

```bash
python -m etl.blender.crop_board --dir path/to/images/ --output_dir cropped/
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
CROP_Y_START = 75   # Top
CROP_Y_END   = 725  # Bottom
CROP_X_START = 75   # Left
CROP_X_END   = 725  # Right
```

## Programmatic Usage

```python
from etl.blender.crop_board import process_single_image, crop_directory

# Crop single image (overwrites original)
process_single_image("synthetic.png")

# Crop to new location with custom name
process_single_image("synthetic.png", output_dir="cropped/", output_name="board.png")

# Crop entire directory
crop_directory("renders/", output_dir="cropped/")
```

## Troubleshooting

### "Cannot read blend file"
- Ensure `blender/chess-set.blend` exists
- Check path in `submission_config.yaml`

### Pieces not positioned correctly
- Verify FEN string format is valid
- Check calibration constants in `generate_synthetic_from_fen.py`

### Crop area wrong
1. Run with `--preview` flag
2. Adjust `CROP_*` constants in `crop_board.py`
3. Re-run preview until rectangle aligns with board
