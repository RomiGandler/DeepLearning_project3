"""
VQGAN Model - Vector Quantized Generative Adversarial Network

This module contains the VQModel used for training on chess dataset.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ..utils import instantiate_from_config
from .modules.diffusionmodules.model import Encoder, Decoder
from .modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(pl.LightningModule):
    """
    Vector Quantized Model (VQGAN).
    
    Encodes images to discrete latent codes via vector quantization,
    trained with reconstruction loss, perceptual loss, and GAN loss.
    """
    
    def __init__(
        self,
        ddconfig,
        lossconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(
            n_embed, embed_dim, beta=0.25,
            remap=remap, sane_index_shape=sane_index_shape
        )
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=None):
        """Load weights from checkpoint."""
        if ignore_keys is None:
            ignore_keys = []
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """Encode image to quantized latent."""
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        """Decode quantized latent to image."""
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None):
        """
        Decode from codebook indices.
        
        Args:
            code_b: Indices tensor. Either:
                - Flat indices (B*H*W,) with shape provided
                - Spatial indices (B, H, W)
            shape: Optional (B, H, W, C) shape for flat indices.
                   If None and code_b is 3D, infers shape automatically.
        
        Returns:
            Decoded image tensor (B, C, H, W)
        """
        if shape is None and len(code_b.shape) == 3:
            # code_b is (B, H, W) spatial indices
            b, h, w = code_b.shape
            shape = (b, h, w, self.quantize.e_dim)
            code_b = code_b.reshape(-1)
        elif shape is None:
            raise ValueError(
                "shape must be provided for flat indices, or pass spatial indices (B, H, W)"
            )
        
        quant_b = self.quantize.get_codebook_entry(code_b, shape)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        """Full forward pass: encode -> quantize -> decode."""
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        """Extract and preprocess input from batch."""
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # Autoencoder update
            aeloss, log_dict_ae = self.loss(
                qloss, x, xrec, optimizer_idx, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # Discriminator update
            discloss, log_dict_disc = self.loss(
                qloss, x, xrec, optimizer_idx, self.global_step,
                last_layer=self.get_last_layer(), split="train"
            )
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        
        aeloss, log_dict_ae = self.loss(
            qloss, x, xrec, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )
        discloss, log_dict_disc = self.loss(
            qloss, x, xrec, 1, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )
        
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return aeloss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        """Generate images for logging."""
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
