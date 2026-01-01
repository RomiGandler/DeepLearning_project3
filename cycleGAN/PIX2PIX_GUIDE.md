# pix2pix Training Guide for Chess Data

## What is pix2pix?

**pix2pix** is a conditional GAN for **paired** image-to-image translation.
- Requires paired training data (input-output correspondence)
- Direct mapping: learns f(A) = B
- Generally produces better results than CycleGAN when you have paired data

**vs CycleGAN**:
- CycleGAN: Unpaired data, learns bidirectional translation
- pix2pix: Paired data, learns direct supervised translation

---

## Step-by-Step Guide

### Step 1: Prepare Your Data

**Important**: pix2pix needs paired images concatenated horizontally.

Each training image should look like: `[Input | Output]`

For chess: `[Synthetic | Real]`

Run the conversion script:
```bash
cd /home/nessm/DeepLearning_project3/cycleGAN
python prepare_pix2pix_data.py
```

This will:
- Take images from `datasets/chess_data/trainA/` (synthetic)
- Take images from `datasets/chess_data/trainB/` (real)
- Create paired images in `datasets/chess_pix2pix/`
- Split into train/val/test (80%/10%/10%)

**Output structure**:
```
datasets/chess_pix2pix/
├── train/
│   ├── 0000.png  # [synthetic|real] side-by-side
│   ├── 0001.png
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

---

### Step 2: Train pix2pix

```bash
cd /home/nessm/DeepLearning_project3/cycleGAN

# Train with wandb logging
python train.py \
  --dataroot ./datasets/chess_pix2pix \
  --name chess_pix2pix \
  --model pix2pix \
  --direction AtoB \
  --use_wandb \
  --netG unet_256 \
  --norm batch \
  --batch_size 1 \
  --n_epochs 100 \
  --n_epochs_decay 100
```

**Key parameters**:
- `--direction AtoB`: synthetic (A) → real (B)
- `--netG unet_256`: U-Net architecture (good for pix2pix)
- `--norm batch`: Batch normalization (standard for pix2pix)
- `--n_epochs`: Total 200 epochs (100 + 100 decay)

---

### Step 3: Monitor Training

While training runs in background:
```bash
cd /home/nessm/DeepLearning_project3/cycleGAN

# View live logs
tail -f training_output.log

# Check generated samples
ls checkpoints/chess_pix2pix/web/images/
```

Or check your wandb dashboard online.

---

### Step 4: Test the Model

After training:
```bash
python test.py \
  --dataroot ./datasets/chess_pix2pix \
  --name chess_pix2pix \
  --model pix2pix \
  --direction AtoB \
  --netG unet_256

# Results saved to:
# results/chess_pix2pix/test_latest/index.html
```

---

## Important Notes

### About Data Pairing

⚠️ **The script randomly pairs synthetic and real images**. This means:
- It doesn't match by chess position (since your data is unpaired)
- The network learns general style transfer, not position-specific mapping

**If you want true paired data**:
1. Ensure each synthetic image has a corresponding real photo of the SAME chess position
2. Name them identically (e.g., `position_001.png` in both trainA and trainB)
3. Modify the script to match by filename instead of random pairing

### When to Use pix2pix vs CycleGAN

**Use pix2pix when**:
✅ You have paired data (same chess position in synthetic and real)
✅ You want supervised learning with direct correspondence
✅ You prioritize accuracy over flexibility

**Use CycleGAN when**:
✅ You have unpaired data (what you have now)
✅ You want to learn general style transfer
✅ You don't have exact correspondence between images

**For your chess project**: Since your data is unpaired, CycleGAN is likely more appropriate. But you can try pix2pix to compare results!

---

## Training Time Estimate

- Similar to CycleGAN: ~200 epochs
- On GTX 1080 Ti: ~3-4 hours for 40 epochs
- Recommended: Train for at least 100 epochs

---

## Quick Comparison

| Aspect | CycleGAN | pix2pix |
|--------|----------|---------|
| Data requirement | Unpaired | Paired |
| Training | Bidirectional | Unidirectional |
| Loss function | Cycle + GAN | Conditional GAN + L1 |
| Quality (paired) | Good | Better |
| Quality (unpaired) | Good | Poor |
| Your situation | ✅ Good fit | ⚠️ Needs pairing |

---

## Running pix2pix in Background

```bash
cd /home/nessm/DeepLearning_project3/cycleGAN

nohup python train.py \
  --dataroot ./datasets/chess_pix2pix \
  --name chess_pix2pix \
  --model pix2pix \
  --direction AtoB \
  --use_wandb \
  --netG unet_256 \
  --norm batch \
  > pix2pix_training.log 2>&1 &

# Monitor with:
tail -f pix2pix_training.log
```

---

## Troubleshooting

### Error: "Expected 4D tensor"
- Check that images are properly concatenated horizontally
- Each image should be 512x256 (256+256 width)

### Error: "Dataset size is 0"
- Check that `datasets/chess_pix2pix/train/` contains .png files
- Verify images are in the correct format

### Poor results
- You may need true paired data (same chess positions)
- Try training longer (200 epochs)
- Adjust `--lambda_L1` (default 100) - controls L1 loss weight

---

## Next Steps

1. **Run the data preparation**: `python prepare_pix2pix_data.py`
2. **Inspect a paired image**: Check if `datasets/chess_pix2pix/train/0000.png` looks correct
3. **Start training**: Use the command above
4. **Compare with CycleGAN**: See which produces better results for your use case

Good luck! 🎯♟️


