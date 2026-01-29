import itertools
import pdb
import random
from PIL import Image
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
import numpy as np
import torch.nn.functional as F
from src.bbdm.model.utils import extract, default

from src.bbdm.model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from src.bbdm.model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from src.bbdm.model.VQGAN.vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class MaskedLatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        self.masked_loss_scale = getattr(model_config.BB.params, 'masked_loss_scale', 0.5)

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond, gt_mask, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
        if gt_mask is not None:
            gt_latent = self.downsample_mask(gt_mask, x_latent.shape[2])
            if gt_latent.shape[1] > 1:
                gt_latent = gt_latent[:, 0:1, :, :]
        context = self.get_cond_stage_context(x_cond)
        return self.super_forward(x_latent.detach(), x_cond_latent.detach(), mask_x=gt_latent.detach(), context=context)
    
    def super_forward(self, x, y, mask_x ,context=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, mask_x, context, t)

    def p_losses(self, x0, y, mask_x, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

        if self.loss_type == 'l1':
            loss_pixel = (objective - objective_recon).abs()
        elif self.loss_type == 'l2':
            loss_pixel = F.mse_loss(objective, objective_recon, reduction='none')
        else:
            raise NotImplementedError()

        # Regular loss (mean over all pixels)
        recloss = loss_pixel.mean()

        # Masked loss
        # Calculate loss only where mask is 1 (turned on)
        if mask_x is not None:
             # Broadcast mask to match channel dimension if necessary
             # mask_x shape: [B, 1, H, W]
             loss_pixel_masked = loss_pixel * mask_x
             
             # Calculate mean loss over the masked region
             mask_sum = mask_x.sum() * loss_pixel.shape[1] # Account for channels
             if mask_sum > 0:
                 masked_recloss = loss_pixel_masked.sum() / mask_sum
             else:
                 masked_recloss = torch.tensor(0.0, device=x0.device)
        else:
             masked_recloss = recloss # Fallback if no mask provided? Or 0? Assuming 0 deviation if no mask or full mask.
             # If no mask is provided, we shouldn't use masked loss. But user architecture implies mask is always there.
             # If mask is None, we probably wouldn't reach here or it would be handled upstream.
             # But let's set it to 0 or recloss.
             # Given the requirement "punishing only in places where the mask is turned on", if no mask, 0 seems appropriate if 'no mask' means 'no region turned on'.
             # But if 'no mask' means 'full image', then recloss.
             # Let's assume mask is always provided as per user setup.
             masked_recloss = torch.tensor(0.0, device=x0.device)

        # Combined loss
        loss = (1.0 - self.masked_loss_scale) * recloss + self.masked_loss_scale * masked_recloss

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": loss,
            "regular_loss": recloss,
            "masked_loss": masked_recloss,
            "x0_recon": x0_recon
        }
        return loss, log_dict

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context
    
    @staticmethod
    def save_image(image, save_path):
        if torch.is_tensor(image):
            image = image.detach().clone()
            image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
            image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(image)
        im.save(save_path)
    
    @staticmethod
    def downsample_mask(mask, target_size):
        """Downsample mask to target size using nearest neighbor."""
        if torch.is_tensor(mask):
            return torch.nn.functional.interpolate(mask, size=(target_size, target_size), mode='nearest')
        
        # Fallback for numpy (original code)
        mask_np = mask
        # mask_np is (H, W) in range [0, 1]
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        mask_downsampled = mask_pil.resize((target_size, target_size), Image.NEAREST)
        return np.array(mask_downsampled) / 255.0

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
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
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        """
        Sample from the model.
        
        Returns:
            If sample_mid_step=False: Final decoded output tensor
            If sample_mid_step=True: List of (step_index, decoded_tensor) tuples
                                     (matching LatentBrownianBridgeModel format)
        """
        x_cond_latent = self.encode(x_cond, cond=True)
        
        if sample_mid_step:
            # Run the full sampling loop in latent space
            temp, _ = self.p_sample_loop(y=x_cond_latent,
                                         context=self.get_cond_stage_context(x_cond),
                                         clip_denoised=clip_denoised,
                                         sample_mid_step=True)

            out_samples = []
            # Define interval - every 20 steps (matching LatentBrownianBridgeModel)
            save_interval = 20

            print(f"Sampling total steps: {len(temp)}. Decoding every {save_interval} steps.")

            # Iterate and decode only the selected frames to save memory/time
            for i in tqdm(range(0, len(temp), save_interval), desc="decoding intermediate steps"):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append((i, out.to('cpu')))

            # Ensure the final clean image is always included
            if (len(temp) - 1) % save_interval != 0:
                with torch.no_grad():
                    out = self.decode(temp[-1].detach(), cond=False)
                out_samples.append((len(temp) - 1, out.to('cpu')))

            return out_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=False)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec
