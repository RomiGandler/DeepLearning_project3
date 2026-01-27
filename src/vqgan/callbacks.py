"""
VQGAN Callbacks - PyTorch Lightning callbacks for logging and visualization.
"""
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class SetupCallback(Callback):
    """Callback to setup logging directories and save configs."""
    
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create directories
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # Save config
            print("Saving project config...")
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, f"{self.now}-config.yaml")
            )


class ImageLogger(Callback):
    """Callback to log reconstruction images during training."""
    
    def __init__(
        self,
        batch_frequency: int = 500,
        max_images: int = 4,
        clamp: bool = True,
        increase_log_steps: bool = True,
        log_on_batch_idx: bool = False,
        log_first_step: bool = True,
    ):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        
        if increase_log_steps:
            self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        else:
            self.log_steps = [self.batch_freq]

    def _should_log(self, batch_idx, global_step):
        """Determine if we should log at this step."""
        if self.log_first_step and global_step == 0:
            return True
        
        check_idx = batch_idx if self.log_on_batch_idx else global_step
        
        if check_idx % self.batch_freq == 0:
            return True
        
        if check_idx in self.log_steps:
            try:
                self.log_steps.remove(check_idx)
            except ValueError:
                pass
            return True
        
        return False

    @rank_zero_only
    def _log_images(self, pl_module, batch, batch_idx, split="train"):
        """Log images to disk and tensorboard."""
        if not hasattr(pl_module, "log_images") or not callable(pl_module.log_images):
            return
        
        if self.max_images <= 0:
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split)

        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

        # Save to disk
        self._save_images_to_disk(
            pl_module, images, batch_idx, split
        )
        
        # Log to tensorboard
        self._log_to_tensorboard(pl_module, images, split)

        if is_train:
            pl_module.train()

    def _save_images_to_disk(self, pl_module, images, batch_idx, split):
        """Save images to disk."""
        root = os.path.join(pl_module.logger.save_dir, "images", split)
        
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1
            grid = grid.permute(1, 2, 0).numpy()  # CHW -> HWC
            grid = (grid * 255).astype(np.uint8)
            
            filename = f"{k}_step-{pl_module.global_step:06d}_batch-{batch_idx:06d}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.fromarray(grid).save(path)

    def _log_to_tensorboard(self, pl_module, images, split):
        """Log images to tensorboard."""
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._should_log(batch_idx, pl_module.global_step):
            self._log_images(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:  # Log first validation batch
            self._log_images(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Log all test batches to save reconstructions for evaluation
        self._log_images(pl_module, batch, batch_idx, split="test")
