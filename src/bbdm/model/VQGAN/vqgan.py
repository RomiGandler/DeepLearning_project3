"""
Inference-only VQGAN for BBDM.

Reuses Encoder/Decoder/Quantizer from src/vqgan but provides a simplified
nn.Module interface (no PyTorch Lightning) suitable for BBDM inference.

BBDM accesses these as separate components:
- encoder(x) -> pre-quant features
- quant_conv(h) -> features ready for quantization
- quantize(h) -> (quant, loss, info)
- decode(quant) -> reconstructed image (includes post_quant_conv internally)

Auto-download: If ckpt_path is None or doesn't exist locally,
automatically downloads from HuggingFace.
"""
import torch
import torch.nn as nn

# Reuse modules from src/vqgan - no code duplication
from src.vqgan.model.modules.diffusionmodules.model import Encoder, Decoder
from src.vqgan.model.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(nn.Module):
    """
    Inference-only VQModel for BBDM.
    
    This is a pure nn.Module (no Lightning dependency) that provides the interface
    expected by BBDM's LatentBrownianBridgeModel:
    
    Attributes accessed by BBDM:
        encoder: Encoder network (callable)
        quant_conv: Conv2d before quantization (callable)
        quantize: VectorQuantizer (callable, returns tuple)
        decode(): Method that decodes quantized latent (includes post_quant_conv)
    
    If ckpt_path is None or file doesn't exist, auto-downloads from HuggingFace.
    """
    
    def __init__(
        self,
        ddconfig,
        lossconfig=None,  # Not used in BBDM inference
        n_embed=16384,
        embed_dim=4,
        ckpt_path=None,
        ignore_keys=None,
        **kwargs  # Accept any additional kwargs for compatibility
    ):
        super().__init__()
        
        if ignore_keys is None:
            ignore_keys = []
        
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.quantize = VectorQuantizer(
            n_embed, embed_dim, beta=0.25
        )
        
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        # Resolve checkpoint path (download from HF if needed)
        resolved_path = self._resolve_ckpt_path(ckpt_path)
        if resolved_path is not None:
            self.init_from_ckpt(resolved_path, ignore_keys=ignore_keys)
    
    def _resolve_ckpt_path(self, ckpt_path):
        """Resolve checkpoint path, downloading from HF if needed."""
        from src.bbdm.checkpoint_utils import resolve_checkpoint
        result = resolve_checkpoint(ckpt_path)
        return str(result) if result else None
    
    def init_from_ckpt(self, path, ignore_keys=None):
        """Load weights from checkpoint."""
        if ignore_keys is None:
            ignore_keys = []
        
        sd = torch.load(path, map_location="cpu", weights_only=False)
        
        # Handle different checkpoint formats
        if "state_dict" in sd:
            sd = sd["state_dict"]
        
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        print(f"Restored VQModel from {path}")
    
    def encode(self, x):
        """Encode image to latent (before quantization)."""
        h = self.encoder(x)
        return h
    
    def decode(self, quant):
        """Decode quantized latent to image."""
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def forward(self, x):
        """Full forward pass: encode -> quant_conv -> quantize -> decode."""
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        dec = self.decode(quant)
        return dec, emb_loss
