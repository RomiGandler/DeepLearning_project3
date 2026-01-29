# Data Module

Dataset handling and HuggingFace resource management.

## Overview

This module handles:
1. Automatic dataset download from HuggingFace
2. Model checkpoint downloads
3. Base dataloader for training

## HuggingFace Resources

### Repository Locations

| Resource | Repository |
|----------|------------|
| Dataset | `roni-hershko/chess_data` |
| Models | `roni-hershko/chess_model` |

### Available Checkpoints

From `roni-hershko/chess_model`:
- `vqgan_f4.ckpt` - VQGAN f4 architecture
- `vqgan_f8.ckpt` - VQGAN f8 architecture
- `latest_model_*.pth` - BBDM checkpoints
- `CelebAMaskHQ-f4.ckpt` - Base VQGAN for fine-tuning
- `CelebAMaskHQ-f16.ckpt` - Base VQGAN f16 for fine-tuning
- `sam3.pt` - SAM model for evaluation

## Usage

### Automatic Download (Recommended)

Most modules auto-download when `dataset_path: null` or checkpoint doesn't exist:

```yaml
# In BBDM config
data:
  dataset_config:
    dataset_path: null  # Auto-downloads from HuggingFace

model:
  VQGAN:
    params:
      ckpt_path: "vqgan_f8.ckpt"  # Auto-downloads if not found
```

### Programmatic Download

```python
from src.data.hf_downloader import HFResourceManager

# Initialize manager
manager = HFResourceManager(
    local_cache_dir="./cache",  # Optional
    token="hf_xxx"              # Optional, for private repos
)

# Download dataset
dataset_path = manager.get_dataset()
print(f"Dataset at: {dataset_path}")

# Download model checkpoint
ckpt_path = manager.get_model_checkpoint("vqgan_f8.ckpt")
print(f"Checkpoint at: {ckpt_path}")

# Force re-download
ckpt_path = manager.get_model_checkpoint("vqgan_f8.ckpt", force_download=True)
```

### Manual Download

```bash
# Using huggingface-cli
pip install huggingface_hub

# Download dataset
huggingface-cli download roni-hershko/chess_data --repo-type dataset --local-dir ./data/dataset

# Download specific model
huggingface-cli download roni-hershko/chess_model vqgan_f8.ckpt --local-dir ./checkpoints
```

## Dataset Structure

```
dataset/
├── train/
│   ├── A/              # Synthetic images (condition)
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── B/              # Real images (target)
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── gt.csv          # FEN annotations
├── val/
│   ├── A/
│   ├── B/
│   └── gt.csv
└── test/
    ├── A/
    ├── B/
    ├── gt.csv
    └── gtfk.csv        # Test annotations
```

### gt.csv Format

```csv
image,fen
0001.png,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
0002.png,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
```

## Base Dataloader

For custom training pipelines:

```python
from src.data.base_dataloader import ChessDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ChessDataset(
    root_dir="./data/dataset/train",
    image_size=256,
    flip=True,  # Random horizontal flip
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

for batch in loader:
    cond_images = batch['A']  # Synthetic images
    target_images = batch['B']  # Real images
    fens = batch['fen']
```

## Environment Variable

For private HuggingFace repos:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

Or pass directly:

```python
manager = HFResourceManager(token="hf_xxxxxxxxxxxxx")
```

## Module Files

| File | Description |
|------|-------------|
| `hf_downloader.py` | HuggingFace download utilities |
| `base_dataloader.py` | PyTorch dataset implementation |
| `dataset/` | Local dataset cache (auto-populated) |
