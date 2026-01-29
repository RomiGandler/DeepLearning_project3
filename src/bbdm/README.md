# BBDM - Brownian Bridge Diffusion Model

Brownian Bridge Diffusion Model for synthetic-to-realistic chess image translation.

## Overview

BBDM operates in VQGAN's latent space to transform synthetic chessboard images into realistic ones. It uses a Brownian Bridge process that directly maps between source and target distributions.

## Architecture Variants

| Config | Latent Size | VQGAN Factor | z_channels | Notes |
|--------|-------------|--------------|------------|-------|
| f4_config.yaml | 64×64 | 4 | 3 | Higher resolution latent |
| f8_config.yaml | 32×32 | 8 | 4 | Balanced (recommended) |
| f16_config.yaml | 16×16 | 16 | 8 | Most compressed |
| masked_config.yaml | 32×32 | 8 | 4 | With mask conditioning |

## Training

### Prerequisites

1. VQGAN checkpoint (auto-downloads from HuggingFace if not found)
2. Dataset (auto-downloads from HuggingFace if `dataset_path: null`)

### Basic Training

```bash
# Train with f4 architecture
python -m src.bbdm.main -c src/bbdm/configs/f4_config.yaml -t

# Train with f8 architecture (recommended)
python -m src.bbdm.main -c src/bbdm/configs/f8_config.yaml -t

# Train with f16 architecture
python -m src.bbdm.main -c src/bbdm/configs/f16_config.yaml -t
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
python -m src.bbdm.main -c src/bbdm/configs/f8_config.yaml -t --gpu_ids 0,1

# Train on 4 GPUs
python -m src.bbdm.main -c src/bbdm/configs/f8_config.yaml -t --gpu_ids 0,1,2,3
```

### Resume Training

```bash
python -m src.bbdm.main -c src/bbdm/configs/f8_config.yaml -t \
    --resume_model checkpoints/latest_model_100.pth \
    --resume_optim checkpoints/latest_optim_sche_100.pth
```

## Evaluation / Sampling

```bash
# Generate samples for evaluation
python -m src.bbdm.main -c src/bbdm/configs/f8_config.yaml --sample_to_eval
```

## Configuration

### Key Config Sections

```yaml
# Training settings
training:
  n_epochs: 300           # Maximum epochs
  n_steps: 200000         # Maximum steps (whichever comes first)
  save_interval: 2        # Save checkpoint every N epochs
  sample_interval: 2      # Generate samples every N epochs
  accumulate_grad_batches: 4

# Data settings
data:
  dataset_config:
    dataset_path: null    # null = auto-download from HuggingFace
    image_size: 256
  train:
    batch_size: 8

# Model settings
model:
  model_load_path: "latest_model_392.pth"  # BBDM checkpoint
  
  VQGAN:
    params:
      ckpt_path: "vqgan_f8.ckpt"  # VQGAN checkpoint (auto-downloads)
```

### Checkpoint Paths

Checkpoints can be specified as:
- **Filename**: `"vqgan_f8.ckpt"` → Auto-downloads from HuggingFace to `./checkpoints/`
- **Full path**: `"/path/to/model.ckpt"` → Uses local file

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

# Initialize pipeline
pipeline = BBDMPipeline(
    config="src/bbdm/configs/f8_config.yaml",
    bbdm_checkpoint="checkpoints/latest_model_392.pth",
    vqgan_checkpoint="checkpoints/vqgan_f8.ckpt",
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
