# Chess Image Generation Pipeline

Generate realistic chessboard images from FEN notation using Brownian Bridge Diffusion Model (BBDM).

## Quick Start

### 1. Clone & Setup Environment

```bash
# Clone repository
git clone https://github.com/RomiGandler/DeepLearning_project3

# Create conda environment
conda env create -f src/environment.yaml
conda activate chess-proj
```

---

## Data & Models

| Resource | Location |
|----------|----------|
| Models | [`roni-hershko/chess_model`](https://huggingface.co/roni-hershko/chess_model) (HuggingFace) |
| Dataset | [`roni-hershko/chess_data`](https://huggingface.co/datasets/roni-hershko/chess_data) (HuggingFace) |
| Full Data Archive | [Google Drive](https://drive.google.com/drive/u/1/folders/1hjfmjmDeAmPB7TRIitV8vqELYtckigy1) |

**Available checkpoints:**  
SAM3 - `sam3.pt`  
VQGAN - `vqgan_f16.ckpt` (***default***), `vqgan_f4.ckpt`, `CelebAMaskHQ-f4.ckpt`, CelebAMaskHQ-f16.ckpt  
BBDM -  `bbdm_f16_mask_guided.pth` (***default***), `bbdm_f16_masked_loss.pth`, `bbdm_f16_mask_guided.pth`, `latest_model_392.pth` (f4)  

### important note
we had a problem uploading the new dataset to HF, meanwhile please upload the data from dataset.zip from https://drive.google.com/drive/u/1/folders/1hjfmjmDeAmPB7TRIitV8vqELYtckigy1 utill we fix this problem.

### Automatic Download

Models and dataset are **automatically downloaded** when needed:

1. **Model checkpoints** - When you specify a filename (e.g., `vqgan_f4.ckpt`) in the config, the code checks if it exists in `./checkpoints/`. If not found, it downloads from `roni-hershko/chess_model` via the `HFResourceManager` in `src/data/hf_downloader.py`.

2. **Dataset** - When `dataset_path: null` in BBDM/VQGAN configs, the dataset auto-downloads from `roni-hershko/chess_data` to `src/data/dataset/`.

3. **SAM model** - The evaluation module auto-downloads `sam3.pt` on first use.

No manual setup required - just run the code and models download automatically.

### Manual Download (Optional)

If you prefer to download manually:
```bash
pip install huggingface_hub

# Download all model checkpoints
huggingface-cli download roni-hershko/chess_model --local-dir ./checkpoints

# Download dataset
huggingface-cli download roni-hershko/chess_data --repo-type dataset --local-dir ./data/dataset
```

---

## Run Inference

### 1. Configure `submission_config.yaml`

```yaml
blender:
  # Path to Blender executable (download from blender.org if needed)
  exec_path: "./blender-5.0.1-linux-x64/blender"
  # Path to the 3D chess set model
  blend_file: "blender/chess-set.blend"
  # Script that generates synthetic images from FEN
  script_file: "etl/blender/generate_synthtic_from_fen.py"

models:
  # BBDM config file (f4, f8, or f16)
  bbdm_config: "src/bbdm/configs/f4_config.yaml"
  # BBDM checkpoint - auto-downloads if just filename
  bbdm_checkpoint: "bbdm_f16_mask_guided.pth"
  # VQGAN checkpoint - auto-downloads if just filename
  vqgan_checkpoint: "vqgan_f16.ckpt"

evaluation:
  # Enable SAM-based piece detection evaluation
  enabled: true
```

**Key settings:**
- `blender.exec_path` - Point to your Blender installation
- `models.*_checkpoint` - Use filename only (auto-downloads) or full path to local file
- `evaluation.enabled` - Set `false` to skip evaluation

### 2. Run

```bash
# Generate from FEN (default: starting position, white view)
python submission.py

# Custom FEN position
python submission.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# From black's perspective
python submission.py --fen "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1" --viewpoint black

# Skip evaluation (faster)
python submission.py --fen "..." --no-eval
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--fen, -f` | FEN string for chess position | Starting position |
| `--viewpoint, -v` | `white` or `black` perspective | `white` |
| `--no-eval` | Skip evaluation step | `false` |

### Output

Results saved to `./results/`:
- `synthetic.png` - Blender-rendered synthetic image
- `realistic.png` - BBDM-generated realistic image
- `side_by_side.png` - Comparison of both
- `predicted_grid.png` - Evaluation grid (if enabled)

---

## Training

### Train VQGAN (Step 1)

```bash
# f4 architecture (recommended)
python -m src.vqgan.main -c src/vqgan/configs/config_train_f4.yaml --epochs 100

# f16 architecture
python -m src.vqgan.main -c src/vqgan/configs/config_train_f16.yaml --epochs 100
```

### Train BBDM (Step 2)

```bash
# f4 config (64×64 latent)
python -m src.bbdm.main -c src/bbdm/configs/f4_config.yaml -t

# f8 config (32×32 latent)


# f16 masked config
python -m src.bbdm.main -c src/bbdm/configs/f16_masked_config.yaml -t

# Multi-GPU training
python -m src.bbdm.main -c src/bbdm/configs/f8_config.yaml -t --gpu_ids 0,1,2,3

# Resume training
python -m src.bbdm.main -c src/bbdm/configs/f4_config.yaml -t \ --resume_model checkpoints/latest_model_392.pth
```

---

## Test and Evaluation (Our Accuracy Score)
First, generate test samples with a model (config) of your choice

```bash
# Evaluate with local dataset
python -m src.bbdm.main -c <your_config> --sample-to-eval
```
in our pipeline, the generated images will be saved by default to results/<dataset name from config> / <model name from config> / samples_to_eval / 200 (but this can be configured).

Once images are generated, run evaluation using:

```bash
# Evaluate with local dataset
python -m src.evaluation.evaluate_model --dataset_path ./data/dataset --stage test --generated_dir <models_results_dir>
```

IMPORTANT NOTE - if you choose to run evaluation of a dataset that isn't straight from the dataset, you'd have to modify the "gt.csv" file being used to include your new FENs and image names.
---

## Project Structure

```
├── submission.py           # Main inference script
├── submission_config.yaml  # Configuration
├── checkpoints/            # Model weights (auto-downloaded)
├── blender/                # Blender project file
├── results/                # Output images
└── src/
    ├── bbdm/               # BBDM diffusion model
    ├── vqgan/              # VQGAN autoencoder
    ├── evaluation/         # SAM-based evaluation
    ├── blender/            # Synthetic image generation
    └── data/               # Dataset utilities & HF downloader
```

---

## Requirements

- Python 3.11
- CUDA 12.1+ (for GPU)
- Blender 5.0+ (for synthetic generation)
- ~8GB GPU memory (inference), ~16GB (training)
