"""BBDM Inference API - simple model loading and generation.

Usage (new checkpoints with embedded config):
    from src.bbdm.inference import load_pipeline
    
    pipe = load_pipeline(
        bbdm_checkpoint="bbdm_f8.pth",
        vqgan_checkpoint="vqgan_f8.ckpt",
    )
    
    result = pipe.generate_from_path("input.png")
    result.save("output.png")

Usage (old checkpoints without embedded config):
    pipe = load_pipeline(
        bbdm_checkpoint="old_model.pth",
        vqgan_checkpoint="vqgan_f8.ckpt",
        config="src/bbdm/configs/f8_config.yaml",  # Required for old checkpoints
    )
"""
import torch
import yaml
from pathlib import Path
from typing import Union, Optional
from PIL import Image
import torchvision.transforms as T

from src.bbdm.checkpoint_utils import resolve_checkpoint, CHECKPOINTS_DIR
from src.bbdm.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from src.bbdm.model.BrownianBridge.MaskedLatentBrownianBridgeModel import MaskedLatentBrownianBridgeModel
from src.bbdm.utils import dict2namespace


class BBDMPipeline:
    """
    Simple inference pipeline for BBDM.
    
    New checkpoints (with embedded config):
        pipe = BBDMPipeline(
            bbdm_checkpoint="bbdm_f8.pth",
            vqgan_checkpoint="vqgan_f8.ckpt",
        )
    
    Old checkpoints (require config file):
        pipe = BBDMPipeline(
            bbdm_checkpoint="old_model.pth",
            vqgan_checkpoint="vqgan_f8.ckpt",
            config="src/bbdm/configs/f8_config.yaml",
        )
    """
    
    def __init__(
        self,
        bbdm_checkpoint: str,
        vqgan_checkpoint: str,
        config: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        checkpoints_dir: Path = CHECKPOINTS_DIR,
    ):
        """
        Initialize BBDM inference pipeline.
        
        Args:
            bbdm_checkpoint: BBDM checkpoint name or path (e.g., "bbdm_f8.pth").
                            Downloads from HF if not found locally.
            vqgan_checkpoint: VQGAN checkpoint name or path (e.g., "vqgan_f8.ckpt").
                             Downloads from HF if not found locally.
            config: Path to config yaml file. Optional for new checkpoints
                   (config embedded), required for old checkpoints.
            device: "cuda" or "cpu".
            checkpoints_dir: Where to look for / download checkpoints.
        """
        self.device = torch.device(device)
        self.checkpoints_dir = Path(checkpoints_dir)
        
        # Resolve and load BBDM checkpoint first to check for embedded config
        bbdm_path = resolve_checkpoint(bbdm_checkpoint, self.checkpoints_dir)
        print(f"Loading BBDM checkpoint from {bbdm_path}...")
        state = torch.load(bbdm_path, map_location="cpu", weights_only=False)
        
        # Get config: from checkpoint (preferred) or from file (fallback)
        self.config = self._load_config(state, config)
        
        # Override VQGAN checkpoint path with user-provided one
        vqgan_path = resolve_checkpoint(vqgan_checkpoint, self.checkpoints_dir)
        self.config.model.VQGAN.params.ckpt_path = str(vqgan_path)
        
        # Initialize model
        print("Initializing model...")
        if hasattr(self.config.model.BB.params, 'masked_loss_scale'):
            self.model = MaskedLatentBrownianBridgeModel(self.config.model)
        else:
            self.model = LatentBrownianBridgeModel(self.config.model)
        
        # Load BBDM weights
        print(f"Loading BBDM weights...")
        self.model.load_state_dict(state["model"], strict=False)
        
        # Load latent statistics if present (for normalize_latent models)
        if 'ori_latent_mean' in state:
            self.model.ori_latent_mean = state['ori_latent_mean'].to(self.device)
            self.model.ori_latent_std = state['ori_latent_std'].to(self.device)
            self.model.cond_latent_mean = state['cond_latent_mean'].to(self.device)
            self.model.cond_latent_std = state['cond_latent_std'].to(self.device)
        
        # Move to device and set eval mode
        self.model = self.model.to(self.device).eval()
        
        # Store preprocessing params
        self.image_size = self.config.data.dataset_config.image_size
        self.to_normal = self.config.data.dataset_config.to_normal
        
        print("Model ready for inference.")
    
    def _load_config(self, checkpoint_state: dict, config_path: Optional[Union[str, Path]]) -> object:
        """
        Load config from checkpoint or file.
        
        Args:
            checkpoint_state: Loaded checkpoint dict
            config_path: Optional path to config file (for old checkpoints)
            
        Returns:
            Config namespace
        """
        # Try embedded config first (new checkpoint format)
        if 'inference_config' in checkpoint_state:
            print("Using config embedded in checkpoint.")
            return dict2namespace(checkpoint_state['inference_config'])
        
        # Fall back to config file (old checkpoint format)
        if config_path is None:
            raise ValueError(
                "This checkpoint doesn't have embedded config. "
                "Please provide a config file path via the 'config' argument. "
                "Example: config='src/bbdm/configs/f8_config.yaml'"
            )
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        print(f"Using config from file: {config_path}")
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        return dict2namespace(config_dict)
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to model input tensor."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])
        tensor = transform(image)
        if self.to_normal:
            tensor = (tensor - 0.5) * 2.0  # [0,1] -> [-1,1]
        return tensor.unsqueeze(0)  # Add batch dim
    
    def _postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """Convert model output tensor to PIL image."""
        tensor = tensor.squeeze(0).cpu()
        if self.to_normal:
            tensor = tensor * 0.5 + 0.5  # [-1,1] -> [0,1]
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).to(torch.uint8)
        return Image.fromarray(tensor.permute(1, 2, 0).numpy())
    
    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """
        Generate from condition tensor.
        
        Args:
            condition: [B, 3, H, W] tensor (should be in [-1,1] if to_normal=True)
            clip_denoised: Whether to clip intermediate samples
            
        Returns:
            Generated images: [B, 3, H, W] tensor
        """
        condition = condition.to(self.device)
        output = self.model.sample(condition, clip_denoised=clip_denoised)
        return output
    
    @torch.no_grad()
    def generate_from_path(
        self,
        image_path: Union[str, Path],
        clip_denoised: bool = False,
    ) -> Image.Image:
        """
        Generate from image file path.
        
        Args:
            image_path: Path to condition image
            clip_denoised: Whether to clip intermediate samples
            
        Returns:
            Generated PIL Image
        """
        image = Image.open(image_path).convert("RGB")
        condition = self._preprocess(image).to(self.device)
        output = self.generate(condition, clip_denoised)
        return self._postprocess(output)


def load_pipeline(
    bbdm_checkpoint: str,
    vqgan_checkpoint: str,
    config: Optional[Union[str, Path]] = None,
    device: str = "cuda",
) -> BBDMPipeline:
    """
    Load BBDM pipeline for inference.
    
    Args:
        bbdm_checkpoint: BBDM checkpoint name or path (e.g., "bbdm_f8.pth").
                        Downloads from HF if not found locally.
        vqgan_checkpoint: VQGAN checkpoint name or path (e.g., "vqgan_f8.ckpt").
                         Downloads from HF if not found locally.
        config: Path to config yaml file. Optional for new checkpoints
               (config embedded), required for old checkpoints.
        device: "cuda" or "cpu".
        
    Returns:
        BBDMPipeline ready for inference.
        
    Example (new checkpoint with embedded config):
        pipe = load_pipeline("bbdm_f8.pth", "vqgan_f8.ckpt")
        result = pipe.generate_from_path("input.png")
        result.save("output.png")
        
    Example (old checkpoint without embedded config):
        pipe = load_pipeline(
            "old_model.pth",
            "vqgan_f8.ckpt",
            config="src/bbdm/configs/f8_config.yaml"
        )
    """
    return BBDMPipeline(
        bbdm_checkpoint=bbdm_checkpoint,
        vqgan_checkpoint=vqgan_checkpoint,
        config=config,
        device=device,
    )
