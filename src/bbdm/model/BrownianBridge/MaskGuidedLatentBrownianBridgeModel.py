"""
Mask-Guided Latent Brownian Bridge Model.

Extends LBBDM with learned mask encoding for white/black piece guidance.
Masks are encoded via SpatialRescaler and concatenated to UNet input.
"""
import itertools
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from src.bbdm.model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from src.bbdm.model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from src.bbdm.model.VQGAN.vqgan import VQModel
from src.bbdm.model.utils import default


def disabled_train(self, mode=True):
    return self


class MaskGuidedLatentBrownianBridgeModel(BrownianBridgeModel):
    """
    LBBDM with mask guidance via learned encoding.
    
    Masks (2ch: white/black) are encoded by SpatialRescaler to latent size,
    then concatenated to x_t before denoising.
    
    UNet in_channels must be: latent_channels + mask_out_channels
    """
    
    def __init__(self, model_config):
        super().__init__(model_config)
        
        # VQGAN for latent encoding (frozen)
        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"Loaded VQGAN from {model_config.VQGAN.params.ckpt_path}")
        
        # Mask encoder: 2ch masks -> latent-sized features
        # n_stages=2 with multiplier=0.5 gives 256->64 for f4
        mask_config = getattr(model_config, 'MaskEncoder', None)
        if mask_config:
            n_stages = mask_config.n_stages
            out_channels = mask_config.out_channels
        else:
            # Default: 2 stages (256->64 for f4), output 2 channels
            n_stages = 2
            out_channels = 2
        
        self.mask_encoder = SpatialRescaler(
            n_stages=n_stages,
            method='bilinear',
            multiplier=0.5,
            in_channels=2,
            out_channels=out_channels,
            bias=True,
        )
        print(f"MaskEncoder: 2ch -> {out_channels}ch, {n_stages} stages (256->{256//(2**n_stages)})")
        
        # Condition stage model (same as LBBDM)
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError(f"Unknown condition_key: {self.condition_key}")
    
    def get_ema_net(self):
        return self
    
    def get_parameters(self):
        """Return trainable parameters: UNet + mask_encoder (+ cond_stage if applicable)."""
        params = list(self.denoise_fn.parameters()) + list(self.mask_encoder.parameters())
        if self.condition_key == 'SpatialRescaler' and self.cond_stage_model is not None:
            params = params + list(self.cond_stage_model.parameters())
            print("Parameters to optimize: UNet, MaskEncoder, SpatialRescaler")
        else:
            print("Parameters to optimize: UNet, MaskEncoder")
        return params
    
    def apply(self, weights_init):
        super().apply(weights_init)
        self.mask_encoder.apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self
    
    def forward(self, x, x_cond, masks_2ch=None, context=None):
        """
        Forward pass with optional mask guidance.
        
        Args:
            x: Target image [B, 3, 256, 256]
            x_cond: Condition image [B, 3, 256, 256]
            masks_2ch: Optional mask guidance [B, 2, 256, 256] (white, black)
            context: Optional pre-computed context
        """
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
        
        # Encode masks if provided
        masks_latent = None
        if masks_2ch is not None:
            masks_latent = self.mask_encoder(masks_2ch)
        
        context = self.get_cond_stage_context(x_cond)
        return self._forward_with_masks(x_latent.detach(), x_cond_latent.detach(), masks_latent, context)
    
    def _forward_with_masks(self, x, y, masks_latent, context=None):
        """Forward in latent space with mask concatenation."""
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        
        b, c, h, w = x.shape
        assert h == self.image_size and w == self.image_size, f'Expected {self.image_size}, got {h}x{w}'
        
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        return self._p_losses_with_masks(x, y, masks_latent, context, t)
    
    def _p_losses_with_masks(self, x0, y, masks_latent, context, t, noise=None):
        """Loss computation with mask-augmented denoising."""
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))
        
        x_t, objective = self.q_sample(x0, y, t, noise)
        
        # Concatenate masks to x_t if provided
        if masks_latent is not None:
            x_t_input = torch.cat([x_t, masks_latent], dim=1)
        else:
            x_t_input = x_t
        
        objective_recon = self.denoise_fn(x_t_input, timesteps=t, context=context)
        
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = torch.nn.functional.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()
        
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        return recloss, {"loss": recloss, "x0_recon": x0_recon}
    
    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context
    
    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        x_latent = self.vqgan.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = self.vqgan.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent
    
    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        if self.model_config.latent_before_quant_conv:
            x_latent = self.vqgan.quant_conv(x_latent)
        x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
        return self.vqgan.decode(x_latent_quant)
    
    @torch.no_grad()
    def _p_sample_with_masks(self, x_t, y, masks_latent, context, i, clip_denoised=False):
        """Single denoising step with mask guidance."""
        b, *_, device = *x_t.shape, x_t.device
        
        # Concatenate masks if provided
        if masks_latent is not None:
            x_t_input = torch.cat([x_t, masks_latent], dim=1)
        else:
            x_t_input = x_t
        
        if self.steps[i] == 0:
            t = torch.full((b,), self.steps[i], device=device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t_input, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((b,), self.steps[i], device=device, dtype=torch.long)
            n_t = torch.full((b,), self.steps[i+1], device=device, dtype=torch.long)
            
            objective_recon = self.denoise_fn(x_t_input, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            
            from src.bbdm.model.utils import extract
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta
            
            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)
            
            return x_tminus_mean + sigma_t * noise, x0_recon
    
    @torch.no_grad()
    def _p_sample_loop_with_masks(self, y, masks_latent, context=None, clip_denoised=True, sample_mid_step=False):
        """Sampling loop with mask guidance."""
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        
        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc='sampling loop'):
                img, x0_recon = self._p_sample_with_masks(imgs[-1], y, masks_latent, context, i, clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc='sampling loop'):
                img, _ = self._p_sample_with_masks(img, y, masks_latent, context, i, clip_denoised)
            return img
    
    @torch.no_grad()
    def sample(self, x_cond, masks_2ch=None, clip_denoised=False, sample_mid_step=False):
        """
        Sample from the model with optional mask guidance.
        
        Args:
            x_cond: Condition image [B, 3, 256, 256]
            masks_2ch: Optional mask guidance [B, 2, 256, 256]
            clip_denoised: Whether to clip intermediate predictions
            sample_mid_step: Whether to return intermediate steps
        """
        x_cond_latent = self.encode(x_cond, cond=True)
        
        # Encode masks if provided
        masks_latent = None
        if masks_2ch is not None:
            masks_latent = self.mask_encoder(masks_2ch)
        
        context = self.get_cond_stage_context(x_cond)
        
        if sample_mid_step:
            temp, _ = self._p_sample_loop_with_masks(
                x_cond_latent, masks_latent, context, clip_denoised, sample_mid_step=True
            )
            
            out_samples = []
            save_interval = 20
            print(f"Sampling {len(temp)} steps. Decoding every {save_interval}.")
            
            for i in tqdm(range(0, len(temp), save_interval), desc="decoding"):
                out = self.decode(temp[i].detach(), cond=False)
                out_samples.append((i, out.cpu()))
            
            if (len(temp) - 1) % save_interval != 0:
                out = self.decode(temp[-1].detach(), cond=False)
                out_samples.append((len(temp) - 1, out.cpu()))
            
            return out_samples
        else:
            x_latent = self._p_sample_loop_with_masks(
                x_cond_latent, masks_latent, context, clip_denoised, sample_mid_step=False
            )
            return self.decode(x_latent, cond=False)
    
    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec
