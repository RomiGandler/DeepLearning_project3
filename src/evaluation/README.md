# Evaluation Module

SAM-based evaluation for chess piece detection and board accuracy metrics.

## Overview

This module evaluates generated chessboard images by:
1. Using SAM (Segment Anything Model) to detect chess pieces
2. Classifying pieces as white or black based on color analysis
3. Comparing detected positions against ground truth FEN
4. Computing accuracy metrics (cell accuracy, F1 scores, etc.)

## Quick Start

### Evaluate Generated Images

```bash
# Evaluate model outputs against test set
python -m src.evaluation.evaluate_model \
    --stage test \
    --generated_dir path/to/generated/images

# Evaluate with local dataset
python -m src.evaluation.evaluate_model \
    --dataset_path /path/to/dataset \
    --stage test \
    --generated_dir path/to/outputs

# Sanity check: evaluate original B/ images (should be ~100% accuracy)
python -m src.evaluation.evaluate_model --stage test
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_path` | Root dataset path (null = auto-download) | None |
| `--stage` | Which split: train, val, test | test |
| `--generated_dir` | Directory with generated images | None |
| `--output_dir` | Output directory for results | Auto |
| `--gt_images_dir` | Optional ground truth images for comparison | None |
| `--no_debug` | Disable debug output saving | False |

## Programmatic Usage

### Evaluate Single Image

```python
import cv2
from src.evaluation.evaluate_model import evaluate_single
from src.evaluation.sam_grid_extractor import SAMGridExtractor
from src.evaluation.data_saver import DataSaver

# Initialize extractor (auto-downloads SAM model)
extractor = SAMGridExtractor()

# Optional: saver for debug outputs
saver = DataSaver("./eval_output")

# Load image
image = cv2.imread("realistic.png")
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Evaluate
metrics = evaluate_single(
    image=image,
    fen=fen,
    file_id="test_image",
    extractor=extractor,
    saver=saver  # Optional
)

print(f"Cell Accuracy: {metrics['cell_accuracy']:.2%}")
print(f"White F1: {metrics['white_f1']:.2%}")
print(f"Black F1: {metrics['black_f1']:.2%}")
```

### Evaluate Batch

```python
from src.evaluation.evaluate_model import evaluate_batch

results = evaluate_batch(
    images=[img1, img2, img3],
    fens=[fen1, fen2, fen3],
    file_ids=["img1", "img2", "img3"],
    extractor=extractor,
    saver=saver
)

# Aggregate metrics
avg_accuracy = sum(r['cell_accuracy'] for r in results) / len(results)
```

## Metrics

| Metric | Description |
|--------|-------------|
| `overall_accuracy` | Percentage of boards with all cells correct |
| `cell_accuracy` | Percentage of individual cells correct |
| `white_accuracy` | Accuracy for white piece detection |
| `black_accuracy` | Accuracy for black piece detection |
| `white_f1` | F1 score for white pieces |
| `black_f1` | F1 score for black pieces |
| `white_precision` | Precision for white pieces |
| `black_precision` | Precision for black pieces |
| `white_recall` | Recall for white pieces |
| `black_recall` | Recall for black pieces |

## Output Structure

```
eval_output/
├── debug/
│   ├── image_001/
│   │   ├── input.png         # Input image
│   │   ├── grid_gt.png       # Ground truth grid visualization
│   │   ├── grid_gt.txt       # Ground truth grid as text
│   │   ├── grid_pred.png     # Predicted grid visualization
│   │   ├── grid_pred.txt     # Predicted grid as text
│   │   ├── detections.png    # SAM detections with annotations
│   │   ├── mask_white.png    # White piece masks
│   │   └── mask_black.png    # Black piece masks
│   └── ...
└── summary.json              # Aggregated metrics
```

## Grid Visualization Colors

In `grid_pred.png` and `grid_gt.png`:
- **Green** - White piece detected
- **Red** - Black piece detected
- **Yellow** - Conflict (both colors detected)
- **Black** - Empty cell

## Module Files

| File | Description |
|------|-------------|
| `evaluate_model.py` | Main evaluation entry point |
| `sam_grid_extractor.py` | SAM-based piece detection |
| `fen_to_grid.py` | FEN to grid conversion utilities |
| `board_metrics.py` | Accuracy and F1 metric computation |
| `data_saver.py` | Debug output saving utilities |
| `dataloader.py` | Dataset loading for batch evaluation |
| `model_loader.py` | SAM model loading utilities |

## SAM Model

The SAM model (`sam3.pt`) is automatically downloaded from HuggingFace when first used. It's cached locally in `./checkpoints/`.
