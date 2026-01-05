# Deep Learning Project 3: Synthetic-to-Real Chess Image Translation

This repository contains the implementation for synthetic-to-real chessboard image translation using **BBDM (Brownian Bridge Diffusion Model)**, **VQGAN (Taming Transformers)**, and **SAM (Segment Anything Model)** for data preprocessing.

---

## 📁 Repository Structure

```
.
├── BBDM/                      # Brownian Bridge Diffusion Model
│   ├── configs/               # YAML configuration files
│   ├── friefeld_data/         # Training data (A=synthetic, B=real) [NOT IN GIT]
│   ├── SAM/                   # Segment Anything Model scripts
│   ├── results/               # Training outputs & checkpoints [NOT IN GIT]
│   └── main.py                # Main training script
├── taming-transformers/       # VQGAN for latent space encoding
│   ├── configs/               # YAML configuration files
│   ├── logs/                  # Training logs & checkpoints [NOT IN GIT]
│   └── main.py                # Main training script
├── cycleGAN/                  # CycleGAN implementation
├── controlNet/                # ControlNet scripts
├── blender/                   # Blender scripts for synthetic data
├── scripts/                   # Utility scripts
├── bbdm_sbatch.sh             # SLURM script for BBDM training
├── run_sbatch.sh              # SLURM script for VQGAN training
└── sam_sbatch.sh              # SLURM script for SAM processing
```

---

## 🔗 Original Repositories

| Model | Repository | Paper |
|-------|-----------|-------|
| **BBDM** | [bo-10000/BBDM](https://github.com/bo-10000/BBDM) | [arXiv:2205.07680](https://arxiv.org/abs/2205.07680) |
| **VQGAN** | [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) | [arXiv:2012.09841](https://arxiv.org/abs/2012.09841) |
| **SAM** | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | Segment Anything Model |

---

## 🛠️ Environment Setup

### 1. BBDM Environment (`roni`)
```bash
conda env create -f BBDM/environment.yml
conda activate roni
```

### 2. VQGAN/Taming Environment (`taming`)
```bash
conda env create -f taming-transformers/environment.yaml
conda activate taming
cd taming-transformers && pip install -e .
```

### 3. SAM Environment (`sam_env`)
```bash
conda create -n sam_env python=3.10
conda activate sam_env
pip install ultralytics opencv-python imagehash pillow
```

---

## 🎯 Pipeline Overview

```
1. SAM → Extract chess piece masks & filter hand images
2. VQGAN → Finetune encoder on chess images (creates latent space)
3. BBDM → Train diffusion model for synthetic→real translation
```

---

## 1️⃣ SAM: Data Preprocessing

### Purpose
SAM is used to:
- **Detect and filter** images containing hands
- **Generate segmentation masks** of chess pieces
- **Deduplicate** similar frames from video data

### Scripts

| Script | Description |
|--------|-------------|
| `BBDM/SAM/extract_chess_pieces.py` | Process `friefeld_data/` - separates hand/no-hand images, generates masks |
| `BBDM/SAM/extract_chess_pieces_pgn.py` | Process `pgn_data/` game folders |
| `BBDM/SAM/deduplicate_frames.py` | Remove duplicate/similar frames using MSE + perceptual hashing |

### Data Structure (Input)
```
friefeld_data/
├── train/
│   ├── A/          # Synthetic images (input)
│   └── B/          # Real images (ground truth)
├── val/
│   ├── A/
│   └── B/
└── test/
    ├── A/
    └── B/
```

### Output Structure
After running SAM:
```
friefeld_data/
├── train/
│   ├── A/          # Original synthetic
│   ├── B/          # Original real
│   ├── no_hand/    # Images without hands ✓
│   ├── with_hand/  # Images with hands (excluded)
│   └── masks/      # Chess piece segmentation masks (PNG)
```

### Running SAM

**Option A: Interactive (on GPU node)**
```bash
srun --gpus=rtx_3090:1 --mem=20G --pty bash
source activate sam_env
cd BBDM/SAM
python extract_chess_pieces.py
```

**Option B: Using SLURM batch script**

Edit `sam_sbatch.sh` to set your script, then:
```bash
bash sam_sbatch.sh
```

**Deduplication (dry run first)**
```bash
python deduplicate_frames.py                    # Dry run - shows what would be deleted
python deduplicate_frames.py --execute          # Actually move duplicates
```

---

## 2️⃣ VQGAN: Finetuning the Latent Encoder

### Purpose
VQGAN learns to encode chess images into a compressed latent space. BBDM operates in this latent space for efficiency.

### Pre-trained Model
Download the base checkpoint and place at `BBDM/results/VQGAN/`:
- **VQGAN-16**: [Download](https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1)

### Data Preparation
Create train/val text files listing image paths:
```bash
cd taming-transformers
find /path/to/your/images -name "*.jpg" | shuf > train.txt
# Split some for validation into val.txt
```

### Configuration
Edit `taming-transformers/configs/chess_finetune.yaml`:
```yaml
model:
  params:
    ckpt_path: "/path/to/pretrained/model.ckpt"
data:
  params:
    train:
      params:
        training_images_list_file: train.txt
    validation:
      params:
        test_images_list_file: val.txt
```

### Running VQGAN Training
```bash
bash run_sbatch.sh
```

**Environment:** `taming`  
**GPU:** RTX 3090  
**Memory:** 40G  

### Output Location
```
taming-transformers/logs/<timestamp>_chess_finetune/
├── checkpoints/           # Model checkpoints (last.ckpt)
├── images/val/            # Reconstruction visualizations
└── configs/               # Saved configuration
```

After training, copy the final checkpoint:
```bash
cp taming-transformers/logs/<timestamp>/checkpoints/last.ckpt BBDM/results/VQGAN/last.ckpt
```

---

## 3️⃣ BBDM: Brownian Bridge Diffusion Model

### Purpose
BBDM performs **image-to-image translation** from synthetic chess images (A) to realistic chess images (B) using diffusion in latent space.

### Data Format
```
friefeld_data/
├── train/
│   ├── A/     # Synthetic images (condition)
│   └── B/     # Real images (target)
├── val/
│   ├── A/
│   └── B/
└── test/
    ├── A/
    └── B/
```
- Images should be **256×256 pixels**
- Paired images must have **matching filenames** in A and B folders

### Configuration
Edit `BBDM/configs/Template-LBBDM-f16.yaml`:

```yaml
data:
  dataset_name: 'your_dataset_name'
  dataset_config:
    dataset_path: 'friefeld_data'    # Relative to BBDM/
    image_size: 256

model:
  VQGAN:
    params:
      ckpt_path: 'results/VQGAN/last.ckpt'  # Your finetuned VQGAN

training:
  n_epochs: 1000
  save_interval: 2
  sample_interval: 2
```

### Running BBDM Training
```bash
bash bbdm_sbatch.sh
```

**Environment:** `roni`  
**GPU:** RTX 3090  
**Memory:** 20G  

### Resume Training
Edit `BBDM/main.py` (uncomment lines 44-45):
```python
args.resume_model = os.path.join(os.getcwd(), 'results/.../checkpoint/latest_model_XXX.pth')
args.resume_optim = os.path.join(os.getcwd(), 'results/.../checkpoint/latest_optim_sche_XXX.pth')
```

### Output Location
```
BBDM/results/<dataset_name>/LBBDM-f16/
├── checkpoint/
│   ├── latest_model_XXX.pth      # Model weights
│   └── latest_optim_sche_XXX.pth # Optimizer state
├── sample/                        # Generated samples during training
└── log/                           # TensorBoard logs
```

### Testing / Inference
```bash
python main.py --config configs/Template-LBBDM-f16.yaml --sample_to_eval --gpu_ids 0 \
    --resume_model path/to/model.pth
```

---

## 📋 SLURM Quick Reference

### Submit a job
```bash
bash bbdm_sbatch.sh    # or run_sbatch.sh, sam_sbatch.sh
```

### Check job status
```bash
squeue -u $USER              # Your running jobs
squeue                       # All jobs
scancel <JOB_ID>             # Cancel a job
```

### View output logs
```bash
tail -f outputs/<job_name>.out      # Follow live output
cat outputs/<job_name>.out          # View full output
```

### Interactive GPU session
```bash
srun --gpus=rtx_3090:1 --mem=20G --time=2:00:00 --pty bash
```

### Check GPU usage
```bash
nvidia-smi
watch -n 1 nvidia-smi       # Auto-refresh every second
```

---

## 🔄 Git Quick Reference

### Clone this repository
```bash
git clone https://github.com/RomiGandler/DeepLearning_project3.git
cd DeepLearning_project3
```

### Pull latest changes
```bash
git pull origin main
```

### Commit and push changes
```bash
git add -A
git status                   # Review what will be committed
git commit -m "Your message"
git push origin main
```

### Check what's ignored
```bash
git status --ignored
```

---

## ⚠️ Important Notes

1. **Data is NOT in git** - Download/prepare data separately (see `.gitignore`)
2. **Model weights are NOT in git** - Download pre-trained models as described above
3. **Always check GPU availability** before submitting jobs
4. **Monitor disk usage** - Training generates large checkpoint files

---

## 📊 Recommended Training Order

1. **Prepare data** using SAM scripts
2. **Finetune VQGAN** on your chess images (~10-20 epochs)
3. **Train BBDM** using the finetuned VQGAN checkpoint

---

## 📧 Contact

For questions about this project, contact the repository maintainers.

---

## 📚 Citations

```bibtex
@inproceedings{li2023bbdm,
  title={BBDM: Image-to-image translation with Brownian bridge diffusion models},
  author={Li, Bo and Xue, Kaitao and Liu, Bin and Lai, Yu-Kun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1952--1961},
  year={2023}
}

@misc{esser2020taming,
  title={Taming Transformers for High-Resolution Image Synthesis}, 
  author={Patrick Esser and Robin Rombach and Björn Ommer},
  year={2020},
  eprint={2012.09841},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

