# VQGAN - Vector Quantized GAN

VQGAN autoencoder for compressing chess images into a discrete latent space.

## Overview

VQGAN encodes 256×256 images into a compressed latent representation using vector quantization. The latent space is then used by BBDM for efficient diffusion-based image translation.

## Architecture Variants

| Config | Downsampling Factor | Latent Size | embed_dim | Codebook Size |
|--------|---------------------|-------------|-----------|---------------|
| config_train_f4.yaml | f4 | 64×64 | 3 | 8192 |
| config_train_f16.yaml | f16 | 16×16 | 8 | 16384 |

## Training

### Prerequisites

- Dataset (auto-downloads from HuggingFace if not specified)
- Optional: Pre-trained checkpoint for fine-tuning

### Basic Training

```bash
# Train f4 model from scratch
python -m src.vqgan.main -c src/vqgan/configs/config_train_f4.yaml

# Train f16 model
python -m src.vqgan.main -c src/vqgan/configs/config_train_f16.yaml

# Train with custom dataset path
python -m src.vqgan.main -c src/vqgan/configs/config_train_f4.yaml -d /path/to/dataset
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-c, --config` | Path to config file | Required |
| `-d, --dataset` | Dataset path (overrides config) | Auto-download |
| `-m, --model` | Checkpoint path to load | None |
| `-o, --output` | Output directory | `logs/vqgan` |
| `--epochs` | Number of training epochs | 100 |
| `--gpus` | Number of GPUs (0 for CPU) | 1 |
| `--batch-size` | Override batch size | Config value |
| `--precision` | Training precision (16 or 32) | 32 |
| `--accumulate-grad` | Gradient accumulation steps | 1 |
| `--test-only` | Run testing only | False |
| `--seed` | Random seed | 42 |

### Fine-tuning from Pre-trained

```bash
# Fine-tune from CelebA checkpoint (specified in config)
python -m src.vqgan.main -c src/vqgan/configs/config_train_f4.yaml

# Or specify checkpoint via command line
python -m src.vqgan.main -c src/vqgan/configs/config_train_f4.yaml -m checkpoints/CelebAMaskHQ-f4.ckpt
```

### Resume Training

```bash
python -m src.vqgan.main -c src/vqgan/configs/config_train.yaml \
    -m logs/vqgan/2024-01-01T12-00-00/checkpoints/last.ckpt
```

### Testing Only

```bash
python -m src.vqgan.main -c src/vqgan/configs/config_test.yaml \
    -m checkpoints/vqgan_f4.ckpt --test-only
```

## Configuration

### Key Config Sections

```yaml
model:
  base_learning_rate: 4.5e-6
  params:
    # Checkpoint - auto-downloads from HuggingFace if not found locally
    ckpt: CelebAMaskHQ-f4.ckpt
    
    # Codebook settings
    embed_dim: 3          # Latent channel dimension
    n_embed: 8192         # Codebook size
    
    # Architecture
    ddconfig:
      z_channels: 3       # Must match embed_dim
      resolution: 256     # Input image resolution
      ch_mult: [1, 2, 4]  # Determines downsampling factor
      
    # Loss configuration
    lossconfig:
      params:
        disc_start: 10000     # Start discriminator after N steps
        disc_weight: 0.8      # Discriminator loss weight
        perceptual_weight: 1.0  # LPIPS loss weight

data:
  params:
    batch_size: 2
    num_workers: 8
    train:
      params:
        size: 256
        image_key: both  # 'A', 'B', or 'both'

training:
  max_epochs: 100
  gpus: 1
  precision: 32
```

### Downsampling Factor

The downsampling factor is determined by `ch_mult`:
- `ch_mult: [1, 2, 4]` → f4 (3 elements = 2 downsampling steps → 256/4 = 64)
- `ch_mult: [1, 2, 2, 4]` → f8 (4 elements = 3 downsampling steps → 256/8 = 32)
- `ch_mult: [1, 1, 2, 2, 4]` → f16 (5 elements = 4 downsampling steps → 256/16 = 16)

## Output Structure

```
logs/vqgan/
└── 2024-01-01T12-00-00/
    ├── configs/
    │   └── config.yaml
    ├── checkpoints/
    │   ├── last.ckpt
    │   └── epoch=50.ckpt
    ├── images/
    │   ├── train/
    │   └── val/
    └── tensorboard/
```

## Inference (Programmatic)

```python
from src.vqgan.model_loader import load_vqgan

# Load model
vqgan = load_vqgan(
    config_path="src/vqgan/configs/config_train.yaml",
    ckpt_path="checkpoints/vqgan_f4.ckpt",
    device="cuda"
)

# Encode image to latent
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image = Image.open("image.png").convert("RGB")
x = transform(image).unsqueeze(0).to("cuda")

# Encode → quantize → decode
z = vqgan.encode(x)
z_q, _, _ = vqgan.quantize(z)
reconstructed = vqgan.decode(z_q)
```

## Pre-trained Checkpoints

Available on HuggingFace (`roni-hershko/chess_model`):
- `vqgan_f4.ckpt` - f4 architecture, fine-tuned on chess
- `vqgan_f8.ckpt` - f8 architecture, fine-tuned on chess
- `CelebAMaskHQ-f4.ckpt` - Base model for f4 fine-tuning
- `CelebAMaskHQ-f16.ckpt` - Base model for f16 fine-tuning
