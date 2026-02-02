# BBDM - Brownian Bridge Diffusion Model

Brownian Bridge Diffusion Model for synthetic-to-realistic chess image translation.

## Overview

BBDM operates in VQGAN's latent space to transform synthetic chessboard images into realistic ones. It uses a Brownian Bridge process that directly maps between source and target distributions.

## Architecture Variants

### By VQGAN Factor (Latent Size)

| Factor | Latent Size | z_channels | n_embed | Notes |
|--------|-------------|------------|---------|-------|
| f4 | 64×64 | 3 | 8192 | Higher resolution latent, more compute |
| f16 | 16×16 | 8 | 16384 | Most compressed, faster training |

### By Training Variant

| Variant | Runner | Dataset Type | Description |
|---------|--------|--------------|-------------|
| Standard | `BBDMRunner` | `custom_aligned` | Basic image-to-image translation |
| Masked Loss | `MaskedBBDMRunner` | `masked_aligned` | Weighted loss focusing on piece regions |
| Mask Guided | `MaskGuidedBBDMRunner` | `mask_guided_aligned` | Learned mask encoding for white/black pieces |

### Available Configs

| Config | Factor | Variant | Purpose |
|--------|--------|---------|---------|
| `f4_config.yaml` | f4 | Standard | Training |
| `f4_masked_config.yaml` | f4 | Masked Loss | Training |
| `f4_mask_guided_config.yaml` | f4 | Mask Guided | Training |
| `f16_config.yaml` | f16 | Standard | Training |
| `f16_test_config.yaml` | f16 | Standard | Testing/Inference |
| `f16_masked_config.yaml` | f16 | Masked Loss | Training |
| `f16_masked_test_config.yaml` | f16 | Masked Loss | Testing/Inference |
| `f16_mask_guided_config.yaml` | f16 | Mask Guided | Training |
| `f16_mask_guided_test_config.yaml` | f16 | Mask Guided | Testing/Inference |

## Training

### Prerequisites

1. VQGAN checkpoint (auto-downloads from HuggingFace if not found)
2. Dataset (auto-downloads from HuggingFace if `dataset_path: null`)

### Basic Training

```bash
# Train standard BBDM with f16 architecture
python -m src.bbdm.main -c src/bbdm/configs/f16_config.yaml -t

# Train standard BBDM with f4 architecture
python -m src.bbdm.main -c src/bbdm/configs/f4_config.yaml -t

# Train with masked loss (weighted loss on piece regions)
python -m src.bbdm.main -c src/bbdm/configs/f16_masked_config.yaml -t

# Train with mask guidance (white/black piece encoding)
python -m src.bbdm.main -c src/bbdm/configs/f16_mask_guided_config.yaml -t
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-c, --config` | Path to config file | **Required** |
| `-t, --train` | Enable training mode | False |
| `-r, --result_path` | Output directory | `results` |
| `-s, --seed` | Random seed | 1234 |
| `--gpu_ids` | GPU IDs (e.g., "0,1,2") | "0" |
| `--resume_model` | Resume from checkpoint | None |
| `--resume_optim` | Resume optimizer state | None |
| `--max_epoch` | Override max epochs | Config value |
| `--max_steps` | Override max steps | Config value |
| `--sample_to_eval` | Run evaluation sampling | False |

### Multi-GPU Training (DDP)

```bash
# Train on GPUs 0 and 1
python -m src.bbdm.main -c src/bbdm/configs/f16_config.yaml -t --gpu_ids 0,1

# Train on 4 GPUs
python -m src.bbdm.main -c src/bbdm/configs/f16_config.yaml -t --gpu_ids 0,1,2,3
```

### Resume Training

```bash
python -m src.bbdm.main -c src/bbdm/configs/f16_config.yaml -t \
    --resume_model checkpoints/latest_model_100.pth \
    --resume_optim checkpoints/latest_optim_sche_100.pth
```

## Evaluation / Sampling

```bash
# Generate samples for evaluation (use test configs)
python -m src.bbdm.main -c src/bbdm/configs/f16_test_config.yaml --sample_to_eval

# For masked loss variant
python -m src.bbdm.main -c src/bbdm/configs/f16_masked_test_config.yaml --sample_to_eval

# For mask guided variant
python -m src.bbdm.main -c src/bbdm/configs/f16_mask_guided_test_config.yaml --sample_to_eval
```

## Configuration

### Key Config Sections

```yaml
# Runner type - determines training behavior
runner: "BBDMRunner"  # or "MaskedBBDMRunner" or "MaskGuidedBBDMRunner"

# Training settings
training:
  n_epochs: 150           # Maximum epochs (400 for f4)
  n_steps: 200000         # Maximum steps (whichever comes first)
  save_interval: 2        # Save checkpoint every N epochs
  sample_interval: 2      # Generate samples every N epochs
  accumulate_grad_batches: 4

# Data settings
data:
  dataset_type: 'custom_aligned'  # or 'masked_aligned' or 'mask_guided_aligned'
  dataset_config:
    dataset_path: null    # null = auto-download from HuggingFace
    image_size: 256
  train:
    batch_size: 30        # f16 (8 for f4 due to larger latent)

# Model settings
model:
  model_load_path: null   # null = train from scratch, or path to checkpoint
  
  VQGAN:
    params:
      ckpt_path: "vqgan_f16.ckpt"  # VQGAN checkpoint (auto-downloads)

  # For Masked Loss variant only:
  BB:
    params:
      masked_loss_scale: 0.75  # Balance between regular and masked loss

  # For Mask Guided variant only:
  MaskEncoder:
    n_stages: 4           # 4 for f16, 2 for f4
    out_channels: 2       # White and black mask channels
```

### Checkpoint Paths

Checkpoints can be specified as:
- **Filename**: `"vqgan_f16.ckpt"` → Auto-downloads from HuggingFace to `./checkpoints/`
- **Full path**: `"/path/to/model.ckpt"` → Uses local file
- **null**: Train from scratch (for BBDM) or download default (for VQGAN)

### Model Checkpoint Naming

BBDM checkpoints are saved as: `bbdm_f{factor}_{variant}.ckpt`
- `bbdm_f16.pth` - Standard BBDM
- `bbdm_f16_masked_loss.pth` - Masked loss variant
- `bbdm_f16_mask_guided.pth` - Mask guided variant

## Output Structure

```
results/
├── logs/                    # TensorBoard logs
├── sample/                  # Generated samples during training
│   ├── epoch_10/
│   │   ├── 0_sample.png
│   │   └── 0_cond.png
│   └── ...
├── latest_model_X.pth       # Model checkpoints
└── latest_optim_sche_X.pth  # Optimizer checkpoints
```

## Inference (Programmatic)

```python
from src.bbdm.inference import BBDMPipeline

# Initialize pipeline (standard BBDM)
pipeline = BBDMPipeline(
    config="src/bbdm/configs/f16_test_config.yaml",
    bbdm_checkpoint="checkpoints/bbdm_f16.ckpt",
    vqgan_checkpoint="checkpoints/vqgan_f16.ckpt",
    device="cuda"
)

# Generate from image path
realistic_image = pipeline.generate_from_path("synthetic.png")
realistic_image.save("realistic.png")

# Generate from PIL Image
from PIL import Image
synthetic = Image.open("synthetic.png")
realistic = pipeline.generate(synthetic)
```

## Dataset Structure

Different variants expect different dataset structures:

### Standard (`custom_aligned`)
```
dataset/
├── train/
│   ├── A/          # Synthetic images (condition)
│   ├── B/          # Real images (target)
│   └── gt.csv      # FEN annotations
├── val/
└── test/
```

### Masked Loss (`masked_aligned`)
```
dataset/
├── train/
│   ├── A/          # Synthetic images
│   ├── B/          # Real images
│   ├── masks/      # Single-channel masks for loss weighting
│   └── gt.csv
├── val/
└── test/
```

### Mask Guided (`mask_guided_aligned`)
```
dataset/
├── train/
│   ├── A/              # Synthetic images
│   ├── B/              # Real images
│   ├── A_mask_white/   # White piece masks
│   ├── A_mask_black/   # Black piece masks
│   └── gt.csv
├── val/
└── test/
```
