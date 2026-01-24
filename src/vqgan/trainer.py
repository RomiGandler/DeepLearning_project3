"""
VQGAN Trainer - Handles training, validation, logging, and checkpointing.

Can be run standalone or imported as a module for external use (e.g., BBDM).

Usage:
    Standalone:
        python -m src.vqgan.trainer --config src/vqgan/config.yaml --dataset /path/to/data
        
    As module:
        from src.vqgan.trainer import VQGANTrainer
        trainer = VQGANTrainer(config_path="src/vqgan/config.yaml")
        trainer.train()
"""
import argparse
import os
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from .dataloader import VQGANChessTrain, VQGANChessVal
from .model.vqgan import VQModel
from .model_loader import VQGANCheckpointLoader


# ============================================================================
# Callbacks for Logging and Image Visualization
# ============================================================================

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


# ============================================================================
# Main Trainer Class
# ============================================================================

class VQGANTrainer:
    """
    VQGAN Trainer class for training, validation, and inference.
    
    Can be used standalone or imported by external modules (e.g., BBDM).
    
    Example:
        # Standalone training
        trainer = VQGANTrainer(config_path="src/vqgan/config.yaml")
        trainer.train(max_epochs=100, gpus=1)
        
        # Use for encoding/decoding (external module)
        trainer = VQGANTrainer(config_path="src/vqgan/config.yaml")
        trainer.setup_model()
        trainer.load_checkpoint("path/to/checkpoint.ckpt")
        latents = trainer.encode(images)
        reconstructed = trainer.decode(latents)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[OmegaConf] = None,
        dataset_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        output_dir: str = "logs/vqgan",
    ):
        """
        Initialize the VQGAN Trainer.
        
        Args:
            config_path: Path to YAML config file
            config: OmegaConf config object (alternative to config_path)
            dataset_path: Override dataset path from config
            checkpoint_path: Path to checkpoint for resuming training
            output_dir: Directory for logs and checkpoints
        """
        # Load config
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = OmegaConf.load(config_path)
        else:
            raise ValueError("Must provide either config_path or config")
        
        self.dataset_path = dataset_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        
        # Will be initialized lazily
        self.model: Optional[VQModel] = None
        self.trainer: Optional[Trainer] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self._device = None

    @property
    def device(self):
        """Get the device the model is on."""
        if self._device is not None:
            return self._device
        if self.model is not None:
            return next(self.model.parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def setup_model(self, auto_download_ckpt: bool = True) -> VQModel:
        """
        Initialize the VQGAN model from config.
        
        Args:
            auto_download_ckpt: If True, download checkpoint from HF if not found locally
        """
        model_config = self.config.model
        
        # Resolve checkpoint (Ultralytics-style: just specify filename)
        # Supports: "vqgan_f4.ckpt", "vqgan_f8.ckpt", "/full/path/model.ckpt", or null
        ckpt = model_config.params.get("ckpt")
        loader = VQGANCheckpointLoader()
        try:
            ckpt_path = loader.resolve(ckpt, auto_download=auto_download_ckpt)
            ckpt_path = str(ckpt_path) if ckpt_path else None
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Training from scratch")
            ckpt_path = None
        
        self.model = VQModel(
            ddconfig=OmegaConf.to_container(model_config.params.ddconfig),
            lossconfig=OmegaConf.to_container(model_config.params.lossconfig),
            n_embed=model_config.params.n_embed,
            embed_dim=model_config.params.embed_dim,
            ckpt_path=ckpt_path,
        )
        
        # Set learning rate based on batch size, number of GPUs, and gradient accumulation
        # lr = base_lr * batch_size * num_gpus * accumulate_grad_batches (linear scaling rule)
        bs = self.config.data.params.batch_size
        base_lr = model_config.base_learning_rate
        ngpu = self.config.training.get("gpus", 1) if hasattr(self.config, "training") else 1
        ngpu = max(1, ngpu)  # Ensure at least 1
        accumulate_grad_batches = self.config.training.get("accumulate_grad_batches", 1) if hasattr(self.config, "training") else 1
        self.model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        
        print(f"Model initialized with learning rate: {self.model.learning_rate:.2e} "
              f"(base_lr={base_lr:.2e} * bs={bs} * ngpu={ngpu} * accum={accumulate_grad_batches})")
        return self.model
    
    def setup_data(self) -> tuple:
        """Create train and validation dataloaders."""
        data_config = self.config.data.params
        
        # Get dataset path
        dataset_path = self.dataset_path
        
        # Get parameters with defaults
        train_size = data_config.train.params.get("size", 256)
        val_size = data_config.validation.params.get("size", 256)
        train_image_key = data_config.train.params.get("image_key", "B")
        val_image_key = data_config.validation.params.get("image_key", "B")
        
        # Create datasets
        train_dataset = VQGANChessTrain(
            size=train_size,
            dataset_path=dataset_path,
            image_key=train_image_key,
        )
        
        val_dataset = VQGANChessVal(
            size=val_size,
            dataset_path=dataset_path,
            image_key=val_image_key,
        )
        
        # Create dataloaders
        batch_size = data_config.batch_size
        num_workers = data_config.get("num_workers", 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        return self.train_loader, self.val_loader
    
    def setup_trainer(
        self,
        max_epochs: int = 100,
        gpus: int = 1,
        precision: int = 32,
        accumulate_grad_batches: int = 1,
        log_every_n_steps: int = 50,
        val_check_interval: float = 1.0,
    ) -> Trainer:
        """
        Setup PyTorch Lightning trainer with callbacks.
        
        Args:
            max_epochs: Maximum number of training epochs
            gpus: Number of GPUs to use (0 for CPU)
            precision: Training precision (16 or 32)
            accumulate_grad_batches: Gradient accumulation steps
            log_every_n_steps: Logging frequency
            val_check_interval: Validation frequency (1.0 = every epoch)
        """
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_dir = self.output_dir / now
        ckpt_dir = log_dir / "checkpoints"
        cfg_dir = log_dir / "configs"
        
        # Callbacks
        callbacks = []
        
        # Setup callback
        setup_callback = SetupCallback(
            resume=self.checkpoint_path is not None,
            now=now,
            logdir=str(log_dir),
            ckptdir=str(ckpt_dir),
            cfgdir=str(cfg_dir),
            config=self.config,
        )
        callbacks.append(setup_callback)
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:06d}-{val/rec_loss:.4f}",
            save_last=True,
            save_top_k=3,
            monitor="val/rec_loss",
            mode="min",
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        
        # Image logger
        image_logger = ImageLogger(
            batch_frequency=500,
            max_images=4,
            clamp=True,
        )
        callbacks.append(image_logger)
        
        # Logger
        logger = TensorBoardLogger(
            save_dir=str(log_dir),
            name="tensorboard",
            version="",
        )
        
        # Create trainer
        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if gpus > 0 else "cpu",
            devices=gpus if gpus > 0 else 1,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            val_check_interval=val_check_interval,
            enable_progress_bar=True,
        )
        
        return self.trainer
    
    def train(
        self,
        max_epochs: int = 100,
        gpus: int = 1,
        **trainer_kwargs,
    ):
        """
        Run training.
        
        Args:
            max_epochs: Maximum number of training epochs
            gpus: Number of GPUs
            **trainer_kwargs: Additional arguments for setup_trainer
        """
        # Setup components
        if self.model is None:
            self.setup_model()
        if self.trainer is None:
            self.setup_trainer(max_epochs=max_epochs, gpus=gpus, **trainer_kwargs)
        if self.train_loader is None:
            self.setup_data()
        
        # Train
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=self.checkpoint_path,
        )
    
    def test(self, checkpoint_path: Optional[str] = None):
        """Run testing/evaluation."""
        if self.model is None:
            self.setup_model()
        if self.trainer is None:
            self.setup_trainer()
        if self.val_loader is None:
            self.setup_data()
        
        ckpt = checkpoint_path or self.checkpoint_path
        self.trainer.test(self.model, dataloaders=self.val_loader, ckpt_path=ckpt)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if self.model is None:
            self.setup_model()
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        print("Checkpoint loaded successfully")
    
    def to(self, device: Union[str, torch.device]):
        """Move model to device."""
        if self.model is None:
            self.setup_model()
        self._device = torch.device(device)
        self.model = self.model.to(self._device)
        return self
    
    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> tuple:
        """
        Encode images to quantized latent space.
        
        Args:
            images: Tensor of shape (B, C, H, W), normalized to [-1, 1]
            
        Returns:
            quant: Quantized latent tensor
            info: Tuple of (perplexity, encodings, encoding_indices)
        """
        if self.model is None:
            self.setup_model()
        
        self.model.eval()
        images = images.to(self.device)
        quant, _, info = self.model.encode(images)
        return quant, info
    
    @torch.no_grad()
    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents back to images.
        
        Args:
            quant: Quantized latent tensor
            
        Returns:
            Reconstructed images, normalized to [-1, 1]
        """
        if self.model is None:
            self.setup_model()
        
        self.model.eval()
        quant = quant.to(self.device)
        return self.model.decode(quant)
    
    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """
        Full encode-decode reconstruction.
        
        Args:
            images: Input images, normalized to [-1, 1]
            
        Returns:
            Reconstructed images
        """
        if self.model is None:
            self.setup_model()
        
        self.model.eval()
        images = images.to(self.device)
        return self.model(images)[0]


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VQGAN Training")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="src/vqgan/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Dataset path (overrides config)"
    )
    parser.add_argument(
        "-r", "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="logs/vqgan",
        help="Output directory for logs and checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (0 for CPU)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run testing only"
    )
    
    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Create trainer
    trainer = VQGANTrainer(
        config_path=args.config,
        dataset_path=args.dataset,
        checkpoint_path=args.resume,
        output_dir=args.output,
    )
    
    # Run
    if args.test_only:
        trainer.test()
    else:
        trainer.train(max_epochs=args.epochs, gpus=args.gpus)


if __name__ == "__main__":
    main()

