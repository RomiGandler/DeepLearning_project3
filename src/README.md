# Chess Image Generation - Source Code

This directory contains all source code for the chess synthetic-to-realistic image generation pipeline.

## Project Overview

The pipeline converts FEN (Forsyth-Edwards Notation) chess positions into realistic chessboard images through:
1. **Blender** - Renders synthetic 3D chessboard images from FEN
2. **VQGAN** - Encodes/decodes images to/from a compressed latent space
3. **BBDM** - Brownian Bridge Diffusion Model transforms synthetic → realistic images
4. **Evaluation** - SAM-based piece detection and accuracy metrics

## Directory Structure

```
src/
├── bbdm/           # BBDM diffusion model for image-to-image translation
├── vqgan/          # VQGAN autoencoder for latent space compression
├── evaluation/     # Model evaluation with SAM-based piece detection
├── blender/        # Blender scripts for synthetic image generation
├── data/           # Dataset handling and HuggingFace downloads
└── environment.yaml # Conda environment specification
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate chess-proj
```

### 2. Run Full Pipeline (submission.py)

```bash
# From project root
python submission.py
```

This will:
- Generate synthetic image from FEN using Blender
- Transform to realistic using BBDM
- Optionally evaluate the output

### 3. Train Models

See individual module READMEs for training instructions:
- [VQGAN Training](vqgan/README.md)
- [BBDM Training](bbdm/README.md)

### 4. Evaluate Model

```bash
python -m src.evaluation.evaluate_model --stage test --generated_dir path/to/outputs
```

## Model Checkpoints

Models are automatically downloaded from HuggingFace (`roni-hershko/chess_model`) when needed:
- `vqgan_f4.ckpt` / `vqgan_f8.ckpt` - VQGAN checkpoints
- `latest_model_*.pth` - BBDM checkpoints

## Dataset

The dataset is hosted on HuggingFace (`roni-hershko/chess_data`) and auto-downloads when training.

Structure:
```
dataset/
├── train/
│   ├── A/          # Synthetic images (condition)
│   ├── B/          # Real images (target)
│   └── gt.csv      # FEN annotations
├── val/
└── test/
```
