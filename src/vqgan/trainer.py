"""
VQGAN Trainer - Core training class for VQGAN model.

Can be imported as a module for external use (e.g., BBDM) or used via CLI (main.py).

Usage as module:
    from src.vqgan.trainer import VQGANTrainer
    
    # Training
    trainer = VQGANTrainer(config_path="src/vqgan/config.yaml")
    trainer.train(max_epochs=100, gpus=1)
    
    # Encoding/Decoding (for BBDM integration)
    trainer = VQGANTrainer(config_path="src/vqgan/config.yaml")
    trainer.setup_model()
    trainer.load_checkpoint("path/to/checkpoint.ckpt")
    latents = trainer.encode(images)
    reconstructed = trainer.decode(latents)
"""
import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .callbacks import SetupCallback, ImageLogger
from .dataloader import VQGANChessTrain, VQGANChessVal, VQGANChessTest
from .model.vqgan import VQModel
from .model_loader import VQGANCheckpointLoader


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
        self.pl_trainer: Optional[Trainer] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
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
        """Create train, validation, and test dataloaders."""
        data_config = self.config.data.params
        
        # Get dataset path
        dataset_path = self.dataset_path
        
        # Get parameters with defaults
        train_size = data_config.train.params.get("size", 256)
        val_size = data_config.validation.params.get("size", 256)
        test_size = data_config.test.params.get("size", 256) if hasattr(data_config, "test") else 256
        train_image_key = data_config.train.params.get("image_key", "both")
        val_image_key = data_config.validation.params.get("image_key", "both")
        test_image_key = data_config.test.params.get("image_key", "both") if hasattr(data_config, "test") else "both"
        
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
        
        test_dataset = VQGANChessTest(
            size=test_size,
            dataset_path=dataset_path,
            image_key=test_image_key,
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
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        return self.train_loader, self.val_loader, self.test_loader
    
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
        self.pl_trainer = Trainer(
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
        
        return self.pl_trainer
    
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
        if self.pl_trainer is None:
            self.setup_trainer(max_epochs=max_epochs, gpus=gpus, **trainer_kwargs)
        if self.train_loader is None:
            self.setup_data()
        
        # Train
        self.pl_trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
            ckpt_path=self.checkpoint_path,
        )
    
    def test(self, checkpoint_path: Optional[str] = None):
        """Run testing/evaluation on test set."""
        if self.model is None:
            self.setup_model()
        if self.pl_trainer is None:
            self.setup_trainer()
        if self.test_loader is None:
            self.setup_data()
        
        ckpt = checkpoint_path or self.checkpoint_path
        self.pl_trainer.test(self.model, dataloaders=self.test_loader, ckpt_path=ckpt)
    
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
