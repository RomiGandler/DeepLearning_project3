# 🎯 Submission Instructions - Chess Image Translation

## 📋 Overview
This project generates realistic chessboard images from synthetic inputs using BBDM (Brownian Bridge Diffusion Model).

**Main Entry Point:** `submission.py`

---

## ⚠️ Important Prerequisites

### Python Version Requirement
**You MUST use Python 3.9, 3.10, or 3.11**

PyTorch does not yet support Python 3.12+. If you have Python 3.13 installed, you need to use an older version.

**Check your Python version:**
```bash
python --version
```

If you see Python 3.12 or 3.13, follow the installation steps below carefully.

---

## 🔧 Installation Guide

### Option 1: Using Conda (Recommended for Easy Python Version Management)

```bash
# Create environment with Python 3.11
conda create -n chess_env python=3.11
conda activate chess_env

# Install PyTorch (adjust for your system)
# For Mac (CPU/MPS):
conda install pytorch torchvision torchaudio -c pytorch

# For Linux/Windows with CUDA:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r submission_requirements.txt
```

### Option 2: Using pyenv + venv (Alternative for Mac/Linux)

```bash
# Install Python 3.11 using pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Create virtual environment
python -m venv chess_env
source chess_env/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r submission_requirements.txt
```

---

## 📦 Required Files & Models

### 1. Blender Installation
Download and install Blender 3.0+ from: https://www.blender.org/download/

### 2. Configure Blender Path
Edit `submission.py` line 20 to match your system:

**macOS:**
```python
BLENDER_EXEC = "/Applications/Blender.app/Contents/MacOS/Blender"
```

**Linux:**
```bash
# Find Blender location
which blender

# Update submission.py with the path
BLENDER_EXEC = "/usr/bin/blender"  # or your path
```

**Windows:**
```python
BLENDER_EXEC = "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe"
```

### 3. Trained Model Files
Ensure these files exist (should be included in submission):

```
results/
├── all_data_f4/
│   └── LBBDM-f4/
│       └── checkpoint/
│           ├── config.yaml
│           └── last_model.pth
└── VQGAN/
    └── last.ckpt
```

---

## ▶️ Running the Code

### Basic Execution
```bash
# Activate environment
conda activate chess_env  # or: source chess_env/bin/activate

# Run submission script
python submission.py
```

### Expected Output
The script will:
1. 🎨 Generate synthetic chessboard from FEN using Blender 
2. ✂️ Crop to board area
3. 🤖 Translate to realistic image using neural network 
4. 🔄 Rotate images if viewpoint is 'black'
5. 🖼️ Create side-by-side comparison

**Output files** (in `./results/`):
- `synthetic.png` - Rendered synthetic chessboard
- `realistic.png` - AI-generated realistic version
- `side_by_side.png` - Comparison image

---

## 🎮 Customization

### Change FEN Position or Viewpoint
Edit `submission.py` lines 204-205:

```python
test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
test_view = "white"  # or "black" for 180° rotation
```

### Example FEN Strings
- Starting position: `"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"`
- Mid-game: `"r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"`
- Endgame: `"8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1"`

---

## 🐛 Troubleshooting

### Issue 1: Python Version Error
```
ERROR: No matching distribution found for torch>=1.12.0
```

**Solution:** You're using Python 3.12+. Use conda to create environment with Python 3.11:
```bash
conda create -n chess_env python=3.11
conda activate chess_env
```

### Issue 2: Blender Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: '/Applications/Blender.app/...'
```

**Solution:** Update the `BLENDER_EXEC` path in `submission.py` line 20, or install Blender.

### Issue 3: Module Import Errors
```
ModuleNotFoundError: No module named 'pytorch_lightning'
```

**Solution:** Reinstall dependencies:
```bash
pip install --upgrade -r submission_requirements.txt
```

### Issue 4: Model Files Not Found
```
FileNotFoundError: results/all_data_f4/LBBDM-f4/checkpoint/last_model.pth
```

**Solution:** Ensure the `results/` folder with trained models is in the project root.

---

## 🔍 Verification Steps

### 1. Check Installation
```bash
python -c "import torch, cv2, yaml, pytorch_lightning; print('✅ All imports successful')"
```

### 2. Check Blender
```bash
# macOS/Linux
/Applications/Blender.app/Contents/MacOS/Blender --version

# Or wherever your Blender is installed
```

### 3. Check Model Files
```bash
ls -lh results/all_data_f4/LBBDM-f4/checkpoint/last_model.pth
ls -lh results/VQGAN/last.ckpt
```
---

## 📁 Project Structure

```
.
├── submission.py                    # Main entry point
├── submission_requirements.txt      # Python dependencies
├── SUBMISSION_INSTRUCTIONS.md       # This file
├── blender/
│   ├── chess-set.blend             # Blender chess scene
│   └── generate_synthtic_from_fen.py
├── scripts/
│   └── crop_board.py               # Board cropping utility
├── BBDM/                           # Model code
│   ├── model/
│   ├── datasets/
│   └── utils.py
└── results/                        # Trained models & outputs
    ├── all_data_f4/LBBDM-f4/
    ├── VQGAN/
    └── [generated images]
```

---
