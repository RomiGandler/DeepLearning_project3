"""
Runner for Mask-Guided BBDM training and inference.
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from PIL import Image

from src.bbdm.Register import Registers
from src.bbdm.runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner
from src.bbdm.model.BrownianBridge.MaskGuidedLatentBrownianBridgeModel import MaskGuidedLatentBrownianBridgeModel
from src.bbdm.runners.utils import weights_init, make_dir, save_single_image, get_dataset, get_image_grid


@Registers.runners.register_with_name('MaskGuidedBBDMRunner')
class MaskGuidedBBDMRunner(BBDMRunner):
    """Runner for mask-guided BBDM with white/black piece masks."""
    
    def __init__(self, config):
        super().__init__(config)
    
    def initialize_model(self, config):
        bbdmnet = MaskGuidedLatentBrownianBridgeModel(config.model).to(config.training.device[0])
        bbdmnet.apply(weights_init)
        return bbdmnet
    
    def get_latent_mean_std(self):
        """Calculate latent statistics, handling 3-tuple batches."""
        train_dataset, _, _ = get_dataset(self.config.data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.train.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )
        
        total_ori_mean, total_cond_mean = None, None
        total_ori_var, total_cond_var = None, None
        
        def calc_mean(batch):
            (x, _), (x_cond, _), _ = batch  # Ignore masks for stats
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            
            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            
            return (
                x_latent.mean(axis=[0, 2, 3], keepdim=True),
                x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            )
        
        def calc_var(batch, ori_mean, cond_mean):
            (x, _), (x_cond, _), _ = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            
            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            
            return (
                ((x_latent - ori_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True),
                ((x_cond_latent - cond_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            )
        
        self.logger("Calculating latent mean...")
        batch_count = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            batch_count += 1
            ori_m, cond_m = calc_mean(batch)
            total_ori_mean = ori_m if total_ori_mean is None else total_ori_mean + ori_m
            total_cond_mean = cond_m if total_cond_mean is None else total_cond_mean + cond_m
        
        self.net.ori_latent_mean = total_ori_mean / batch_count
        self.net.cond_latent_mean = total_cond_mean / batch_count
        
        self.logger("Calculating latent std...")
        batch_count = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            batch_count += 1
            ori_v, cond_v = calc_var(batch, self.net.ori_latent_mean, self.net.cond_latent_mean)
            total_ori_var = ori_v if total_ori_var is None else total_ori_var + ori_v
            total_cond_var = cond_v if total_cond_var is None else total_cond_var + cond_v
        
        self.net.ori_latent_std = torch.sqrt(total_ori_var / batch_count)
        self.net.cond_latent_std = torch.sqrt(total_cond_var / batch_count)
        
        self.logger(f"ori_latent_mean: {self.net.ori_latent_mean}")
        self.logger(f"ori_latent_std: {self.net.ori_latent_std}")
    
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        """Loss with mask guidance."""
        (x, _), (x_cond, _), (masks_2ch, _) = batch
        
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])
        masks_2ch = masks_2ch.to(self.config.training.device[0])
        
        loss, info = net(x, x_cond, masks_2ch)
        
        if write and self.is_main_process:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
        
        return loss
    
    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        """Sample with mask guidance."""
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        progress_path = make_dir(os.path.join(sample_path, f'{stage}_progress'))
        
        (x, _), (x_cond, _), (masks_2ch, _) = batch
        
        batch_size = min(x.shape[0], 4)
        x = x[:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[:batch_size].to(self.config.training.device[0])
        masks_2ch = masks_2ch[:batch_size].to(self.config.training.device[0])
        
        grid_size = 4
        to_normal = self.config.data.dataset_config.to_normal
        
        is_validation = stage != 'train'
        
        if is_validation:
            samples_list = net.sample(x_cond, masks_2ch, clip_denoised=self.config.testing.clip_denoised, sample_mid_step=True)
            
            for step_idx, sample_tensor in samples_list:
                image_grid = get_image_grid(sample_tensor, grid_size, to_normal=to_normal)
                im = Image.fromarray(image_grid)
                im.save(os.path.join(progress_path, f"progress_step_{step_idx:04d}.png"))
            
            sample = samples_list[-1][1]
        else:
            sample = net.sample(x_cond, masks_2ch, clip_denoised=self.config.testing.clip_denoised, sample_mid_step=False).cpu()
        
        # Save sample
        image_grid = get_image_grid(sample, grid_size, to_normal=to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')
        
        # Save condition
        image_grid = get_image_grid(x_cond.cpu(), grid_size, to_normal=to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')
        
        # Save ground truth
        image_grid = get_image_grid(x.cpu(), grid_size, to_normal=to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
        
        # Save mask visualization (sum both channels for visibility)
        mask_viz = (masks_2ch[:, 0] + masks_2ch[:, 1]).unsqueeze(1).repeat(1, 3, 1, 1).cpu()
        image_grid = get_image_grid(mask_viz, grid_size, to_normal=False)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'masks.png'))
    
    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        """Evaluation with mask guidance."""
        condition_path = make_dir(os.path.join(sample_path, 'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))
        
        pbar = tqdm(test_loader, total=len(test_loader))
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        
        for test_batch in pbar:
            (x, x_name), (x_cond, x_cond_name), (masks_2ch, _) = test_batch
            
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])
            masks_2ch = masks_2ch.to(self.config.training.device[0])
            
            for j in range(sample_num):
                sample = net.sample(x_cond, masks_2ch, clip_denoised=False)
                
                for i in range(batch_size):
                    if i >= x.shape[0]:
                        break
                    
                    condition = x_cond[i].detach().clone()
                    gt = x[i]
                    result = sample[i]
                    
                    if j == 0:
                        save_single_image(condition, condition_path, f'{x_cond_name[i]}.png', to_normal=to_normal)
                        save_single_image(gt, gt_path, f'{x_name[i]}.png', to_normal=to_normal)
                    
                    if sample_num > 1:
                        result_path_i = make_dir(os.path.join(result_path, x_name[i]))
                        save_single_image(result, result_path_i, f'output_{j}.png', to_normal=to_normal)
                    else:
                        save_single_image(result, result_path, f'{x_name[i]}.png', to_normal=to_normal)
