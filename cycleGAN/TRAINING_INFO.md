# CycleGAN Training Status

## ✅ Training is RUNNING Successfully on GPU

### Setup Summary
- **Model**: CycleGAN
- **Dataset**: chess_data (944 training images)
- **Device**: NVIDIA GeForce GTX 1080 Ti (GPU)
- **PyTorch Version**: 1.13.1+cu116 (compatible with GTX 1080 Ti)
- **Epochs**: 200 (100 with initial learning rate + 100 with decay)
- **Batch Size**: 1

### Where Everything is Saved

#### 1. Model Checkpoints
**Location**: `/home/nessm/DeepLearning_project3/cycleGAN/checkpoints/chess_data_cyclegan/`

- **Automatic saves**:
  - Every 5,000 iterations: `latest_net_G_A.pth`, `latest_net_G_B.pth`, `latest_net_D_A.pth`, `latest_net_D_B.pth`
  - Every 5 epochs: `5_net_G_A.pth`, `10_net_G_A.pth`, etc.
  
- These files will persist even if training stops unexpectedly
- You can resume training with: `--continue_train --epoch_count N`

#### 2. Generated Images
**Location**: `/home/nessm/DeepLearning_project3/cycleGAN/checkpoints/chess_data_cyclegan/web/images/`

- Saves visualization images showing:
  - `real_A` and `real_B`: Original images from both domains
  - `fake_A` and `fake_B`: Generated translations
  - `rec_A` and `rec_B`: Cycle-reconstructed images
  - `idt_A` and `idt_B`: Identity mappings

#### 3. Training Logs
- **Console output**: `/home/nessm/DeepLearning_project3/cycleGAN/training_output.log`
- **Loss log**: `/home/nessm/DeepLearning_project3/cycleGAN/checkpoints/chess_data_cyclegan/loss_log.txt`
- **Training options**: `/home/nessm/DeepLearning_project3/cycleGAN/checkpoints/chess_data_cyclegan/train_opt.txt`

#### 4. Weights & Biases (wandb)
**Online Dashboard**: https://wandb.ai/michalness1-ben-gurion-university-of-the-negev/CycleGAN-and-pix2pix/runs/mctfyucw

- Real-time metrics and loss curves
- Image samples uploaded during training
- Local backup: `/home/nessm/DeepLearning_project3/cycleGAN/wandb/`

### Training Parameters
- **Generator networks (G_A, G_B)**: 11.378M parameters each (ResNet with 9 blocks)
- **Discriminator networks (D_A, D_B)**: 2.765M parameters each (PatchGAN)
- **Learning rate**: 0.0002 (linear decay after epoch 100)
- **Loss weights**:
  - Cycle consistency (A→B→A, B→A→B): λ = 10.0
  - Identity loss: λ = 0.5
  - GAN loss: LSGAN (Least Squares GAN)

### Monitoring Commands

#### Quick Status Check
```bash
cd /home/nessm/DeepLearning_project3/cycleGAN
./monitor_training.sh
```

#### View Live Training Logs
```bash
cd /home/nessm/DeepLearning_project3/cycleGAN
tail -f training_output.log
```

#### Check if Training is Running
```bash
ps aux | grep train.py | grep -v grep
```

#### Stop Training (if needed)
```bash
pkill -f train.py
```

#### GPU Usage
```bash
nvidia-smi
```

### Current Progress
- **Epoch 1/200** completed (279 seconds)
- Training losses are decreasing normally
- Images being generated and saved successfully
- All checkpoints will be automatically saved

### Data Persistence Guarantees
✅ All model checkpoints are saved to disk every 5,000 iterations and every 5 epochs
✅ Training logs are continuously written to files
✅ Generated images are saved regularly
✅ wandb data is both synced online AND saved locally
✅ If training stops for any reason, you can resume from the latest checkpoint

### Expected Training Time
- Epoch 1 took ~279 seconds (~4.6 minutes)
- Estimated total time: 200 epochs × 4.6 min ≈ **15-16 hours**
- The training will run continuously in the background until completion

### Next Steps
The training will continue running in the background. You can:
1. Monitor progress via the wandb dashboard (link above)
2. Run `./monitor_training.sh` periodically to check status
3. Let it run overnight - it's running on GPU and will complete automatically
4. After training completes, find your trained models in the checkpoints directory

---
**Training started**: Dec 31, 2025 10:42 AM
**Process ID**: 475288

