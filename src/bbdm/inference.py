"""BBDM Inference API - simple model loading and generation.

Usage (new checkpoints - self-contained):
    from src.bbdm.inference import load_pipeline
    
    pipe = load_pipeline("bbdm_f8.pth")
    result = pipe.generate_from_path("input.png")
    result.save("output.png")

Usage (old checkpoints - need separate VQGAN):
    pipe = load_pipeline(
        bbdm_checkpoint="old_model.pth",
        vqgan_checkpoint="vqgan_f8.ckpt",
        config="src/bbdm/configs/f8_config.yaml",
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
from src.bbdm.model.BrownianBridge.MaskGuidedLatentBrownianBridgeModel import MaskGuidedLatentBrownianBridgeModel
from src.bbdm.utils import dict2namespace


class BBDMPipeline:
    """
    Simple inference pipeline for BBDM.
    
    New checkpoints (self-contained with VQGAN weights):
        pipe = BBDMPipeline(bbdm_checkpoint="bbdm_f8.pth")
    
    Old checkpoints (require separate VQGAN):
        pipe = BBDMPipeline(
            bbdm_checkpoint="old_model.pth",
            vqgan_checkpoint="vqgan_f8.ckpt",
            config="src/bbdm/configs/f8_config.yaml",
        )
    """
    
    def __init__(
        self,
        bbdm_checkpoint: str,
        vqgan_checkpoint: Optional[str] = None,
        config: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        checkpoints_dir: Path = CHECKPOINTS_DIR,
    ):
        """
        Initialize BBDM inference pipeline.
        
        Args:
            bbdm_checkpoint: BBDM checkpoint name or path (e.g., "bbdm_f8.pth").
                            Downloads from HF if not found locally.
            vqgan_checkpoint: Optional VQGAN checkpoint. Not needed for new checkpoints
                             (VQGAN weights included). Required for old checkpoints.
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
        
        # Handle VQGAN: use provided path, or rely on checkpoint containing VQGAN weights
        has_vqgan_in_checkpoint = any(k.startswith('vqgan.') for k in state['model'].keys())
        
        if vqgan_checkpoint is not None:
            # User provided explicit VQGAN - use it
            vqgan_path = resolve_checkpoint(vqgan_checkpoint, self.checkpoints_dir)
            self.config.model.VQGAN.params.ckpt_path = str(vqgan_path)
        elif has_vqgan_in_checkpoint:
            # VQGAN weights in BBDM checkpoint - don't load separately
            self.config.model.VQGAN.params.ckpt_path = None
        else:
            # Old checkpoint without VQGAN - need separate file
            raise ValueError(
                "This BBDM checkpoint doesn't contain VQGAN weights. "
                "Please provide vqgan_checkpoint parameter."
            )
        
        # Initialize model based on config type
        print("Initializing model...")
        if hasattr(self.config.model, 'MaskEncoder'):
            self.model = MaskGuidedLatentBrownianBridgeModel(self.config.model)
            self.is_mask_guided = True
            print("Loaded mask-guided model (requires masks during inference)")
        elif hasattr(self.config.model.BB.params, 'masked_loss_scale'):
            self.model = MaskedLatentBrownianBridgeModel(self.config.model)
            self.is_mask_guided = False
        else:
            self.model = LatentBrownianBridgeModel(self.config.model)
            self.is_mask_guided = False
        
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
        masks: Optional[torch.Tensor] = None,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """
        Generate from condition tensor.
        
        Args:
            condition: [B, 3, H, W] tensor (should be in [-1,1] if to_normal=True)
            masks: [B, 2, H, W] tensor for mask-guided models (white mask ch0, black mask ch1)
            clip_denoised: Whether to clip intermediate samples
            
        Returns:
            Generated images: [B, 3, H, W] tensor
        """
        # Validate mask usage
        if self.is_mask_guided and masks is None:
            raise ValueError(
                "This is a mask-guided model. You must provide masks during inference. "
                "Pass masks=[B, 2, H, W] tensor with white piece mask in channel 0, black in channel 1."
            )
        if not self.is_mask_guided and masks is not None:
            raise ValueError(
                "This model was not trained with mask guidance. "
                "Do not provide masks during inference."
            )
        
        condition = condition.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
            output = self.model.sample(condition, masks, clip_denoised=clip_denoised)
        else:
            output = self.model.sample(condition, clip_denoised=clip_denoised)
        return output
    
    @torch.no_grad()
    def generate_from_path(
        self,
        image_path: Union[str, Path],
        masks: Optional[torch.Tensor] = None,
        clip_denoised: bool = False,
    ) -> Image.Image:
        """
        Generate from image file path.
        
        Args:
            image_path: Path to condition image
            masks: Optional [1, 2, H, W] tensor for mask-guided models
            clip_denoised: Whether to clip intermediate samples
            
        Returns:
            Generated PIL Image
        """
        image = Image.open(image_path).convert("RGB")
        condition = self._preprocess(image).to(self.device)
        
        # Resize masks to match image_size if provided
        if masks is not None:
            if masks.dim() == 3:
                masks = masks.unsqueeze(0)
            if masks.shape[-1] != self.image_size:
                masks = torch.nn.functional.interpolate(
                    masks.float(), size=(self.image_size, self.image_size), mode='nearest'
                )
            masks = masks.to(self.device)
        
        output = self.generate(condition, masks=masks, clip_denoised=clip_denoised)
        return self._postprocess(output)


def load_pipeline(
    bbdm_checkpoint: str,
    vqgan_checkpoint: Optional[str] = None,
    config: Optional[Union[str, Path]] = None,
    device: str = "cuda",
) -> BBDMPipeline:
    """
    Load BBDM pipeline for inference.
    
    Args:
        bbdm_checkpoint: BBDM checkpoint name or path (e.g., "bbdm_f8.pth").
                        Downloads from HF if not found locally.
        vqgan_checkpoint: Optional VQGAN checkpoint. Not needed for new checkpoints
                         (VQGAN weights included). Required for old checkpoints.
        config: Path to config yaml file. Optional for new checkpoints
               (config embedded), required for old checkpoints.
        device: "cuda" or "cpu".
        
    Returns:
        BBDMPipeline ready for inference.
        
    Example (new checkpoint - self-contained):
        pipe = load_pipeline("bbdm_f8.pth")
        result = pipe.generate_from_path("input.png")
        result.save("output.png")
        
    Example (old checkpoint - needs VQGAN):
        pipe = load_pipeline(
            "old_model.pth",
            vqgan_checkpoint="vqgan_f8.ckpt",
            config="src/bbdm/configs/f8_config.yaml"
        )
    """
    return BBDMPipeline(
        bbdm_checkpoint=bbdm_checkpoint,
        vqgan_checkpoint=vqgan_checkpoint,
        config=config,
        device=device,
    )
