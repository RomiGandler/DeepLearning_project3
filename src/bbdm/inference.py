"""BBDM Inference API - simple model loading and generation.

Usage:
    from src.bbdm.inference import load_pipeline
    
    pipe = load_pipeline(
        config="src/bbdm/configs/f8_config.yaml",
        bbdm_checkpoint="bbdm_f8.pth",
        vqgan_checkpoint="vqgan_f8.ckpt",
    )
    
    # Single image inference
    result = pipe.generate_from_path("input.png")
    result.save("output.png")
    
    # Batch inference
    outputs = pipe.generate(batch_tensor)
"""
import torch
import yaml
from pathlib import Path
from typing import Union
from PIL import Image
import torchvision.transforms as T

from src.bbdm.checkpoint_utils import resolve_checkpoint, CHECKPOINTS_DIR
from src.bbdm.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from src.bbdm.model.BrownianBridge.MaskedLatentBrownianBridgeModel import MaskedLatentBrownianBridgeModel
from src.bbdm.utils import dict2namespace


class BBDMPipeline:
    """
    Simple inference pipeline for BBDM.
    
    All parameters are required - no defaults. User must explicitly provide:
    - config: Path to config yaml file
    - bbdm_checkpoint: BBDM checkpoint name or path
    - vqgan_checkpoint: VQGAN checkpoint name or path
    
    Example:
        pipe = BBDMPipeline(
            config="src/bbdm/configs/f8_config.yaml",
            bbdm_checkpoint="bbdm_f8.pth",
            vqgan_checkpoint="vqgan_f8.ckpt",
        )
        
        # Generate from tensor
        output = pipe.generate(condition_tensor)
        
        # Generate from image path
        output_pil = pipe.generate_from_path("input.png")
    """
    
    def __init__(
        self,
        config: Union[str, Path],
        bbdm_checkpoint: str,
        vqgan_checkpoint: str,
        device: str = "cuda",
        checkpoints_dir: Path = CHECKPOINTS_DIR,
    ):
        """
        Initialize BBDM inference pipeline.
        
        Args:
            config: Path to config yaml file (required).
            bbdm_checkpoint: BBDM checkpoint name or path (e.g., "bbdm_f8.pth").
                            Required - will download from HF if not found locally.
            vqgan_checkpoint: VQGAN checkpoint name or path (e.g., "vqgan_f8.ckpt").
                             Required - will download from HF if not found locally.
            device: "cuda" or "cpu".
            checkpoints_dir: Where to look for / download checkpoints.
        """
        self.device = torch.device(device)
        self.checkpoints_dir = Path(checkpoints_dir)
        
        # Load config
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        self.config = dict2namespace(config_dict)
        
        # Resolve VQGAN checkpoint (required)
        vqgan_path = resolve_checkpoint(vqgan_checkpoint, self.checkpoints_dir)
        self.config.model.VQGAN.params.ckpt_path = str(vqgan_path)
        
        # Initialize model (VQGAN is loaded inside)
        print("Initializing model...")
        if hasattr(self.config.model.BB.params, 'masked_loss_scale'):
            self.model = MaskedLatentBrownianBridgeModel(self.config.model)
        else:
            self.model = LatentBrownianBridgeModel(self.config.model)
        
        # Resolve and load BBDM weights (required)
        bbdm_path = resolve_checkpoint(bbdm_checkpoint, self.checkpoints_dir)
        print(f"Loading BBDM weights from {bbdm_path}...")
        state = torch.load(bbdm_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=False)
        
        # Move to device and set eval mode
        self.model = self.model.to(self.device).eval()
        
        # Store image size from config
        self.image_size = self.config.data.dataset_config.image_size
        self.to_normal = self.config.data.dataset_config.to_normal
        
        print("Model ready for inference.")
    
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
    config: Union[str, Path],
    bbdm_checkpoint: str,
    vqgan_checkpoint: str,
    device: str = "cuda",
) -> BBDMPipeline:
    """
    Load BBDM pipeline for inference.
    
    Args:
        config: Path to config yaml file (required).
        bbdm_checkpoint: BBDM checkpoint name or path (e.g., "bbdm_f8.pth").
                        Required - downloads from HF if not found locally.
        vqgan_checkpoint: VQGAN checkpoint name or path (e.g., "vqgan_f8.ckpt").
                         Required - downloads from HF if not found locally.
        device: "cuda" or "cpu".
        
    Returns:
        BBDMPipeline ready for inference.
        
    Example:
        pipe = load_pipeline(
            config="src/bbdm/configs/f8_config.yaml",
            bbdm_checkpoint="bbdm_f8.pth",
            vqgan_checkpoint="vqgan_f8.ckpt",
        )
        result = pipe.generate_from_path("input.png")
        result.save("output.png")
    """
    return BBDMPipeline(
        config=config,
        bbdm_checkpoint=bbdm_checkpoint,
        vqgan_checkpoint=vqgan_checkpoint,
        device=device,
    )
