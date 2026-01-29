import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from src.bbdm.Register import Registers
from src.bbdm.runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner
from src.bbdm.model.BrownianBridge.MaskedLatentBrownianBridgeModel import MaskedLatentBrownianBridgeModel
from src.bbdm.runners.utils import weights_init, make_dir, save_single_image, get_dataset

@Registers.runners.register_with_name('MaskedBBDMRunner')
class MaskedBBDMRunner(BBDMRunner):
    def __init__(self, config):
        super().__init__(config)
        
    def initialize_model(self, config):
        # Always use MaskedLatentBrownianBridgeModel for this runner
        bbdmnet = MaskedLatentBrownianBridgeModel(config.model).to(config.training.device[0])
        bbdmnet.apply(weights_init)
        return bbdmnet
    
    @staticmethod
    def save_image(image, save_path):
        if torch.is_tensor(image):
            image = image.detach().clone()
            image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
            image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(image)
        im.save(save_path)

    def get_latent_mean_std(self):
        """
        Override to handle 3-tuple batches from MaskedAlignedDataset.
        """
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            # Handle 3-tuple batch: (x, x_cond, mask)
            (x, x_name), (x_cond, x_cond_name), _ = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            # Handle 3-tuple batch: (x, x_cond, mask)
            (x, x_name), (x_cond, x_cond_name), _ = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        self.logger(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        self.logger(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        self.logger(self.net.ori_latent_mean)
        self.logger(self.net.ori_latent_std)
        self.logger(self.net.cond_latent_mean)
        self.logger(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        # Handle 3-tuple batch from MaskedAlignedDataset
        (x, x_name), (x_cond, x_cond_name), (mask, mask_name) = batch

        mask = mask.to(self.config.training.device[0])
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        # Pass gt_mask to forward
        loss, additional_info = net(x, x_cond, gt_mask=mask)
        
        if write and self.is_main_process:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if 'recloss_noise' in additional_info:
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if 'recloss_xy' in additional_info:
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
            if 'regular_loss' in additional_info:
                self.writer.add_scalar(f'regular_loss/{stage}', additional_info['regular_loss'], step)
            if 'masked_loss' in additional_info:
                self.writer.add_scalar(f'masked_loss/{stage}', additional_info['masked_loss'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        # Override sample to ignore mask for sampling/inference
        # Extract x and x_cond, ignoring the mask which is the 3rd element
        (x, x_name), (x_cond, x_cond_name), _ = batch
        
        # Reconstruct batch as 2-tuple expected by parent class
        new_batch = ((x, x_name), (x_cond, x_cond_name))
        
        # Call the parent sample method
        super().sample(net, new_batch, sample_path, stage)

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        """
        Override to handle 3-tuple batch format from MaskedAlignedDataset.
        
        Masks are only used during training loss computation, not during inference.
        """
        condition_path = make_dir(os.path.join(sample_path, 'condition'))
        gt_path = make_dir(os.path.join(sample_path, 'ground_truth'))
        result_path = make_dir(os.path.join(sample_path, str(self.config.model.BB.params.sample_step)))

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        batch_size = self.config.data.test.batch_size
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        
        for test_batch in pbar:
            # Handle 3-tuple batch: extract x and x_cond, ignore mask
            (x, x_name), (x_cond, x_cond_name), _ = test_batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            for j in range(sample_num):
                sample = net.sample(x_cond, clip_denoised=False)
                for i in range(batch_size):
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
