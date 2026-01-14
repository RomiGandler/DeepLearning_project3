from PIL import Image
import torch
from Register import Registers
from runners.DiffusionBasedModelRunners.BBDMRunner import BBDMRunner
from model.BrownianBridge.MaskedLatentBrownianBridgeModel import MaskedLatentBrownianBridgeModel
from runners.utils import weights_init

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
        
        # Reconstruct batch as expected by the parent class (or just pass what's needed)
        # BBDMRunner.sample expects a batch that can be unpacked, but since we are calling it
        # we might need to repack it or call the net.sample directly like BBDMRunner does.
        # BBDMRunner.sample method unpacks: (x, x_name), (x_cond, x_cond_name) = batch
        
        # So we create a new batch tuple without the mask
        new_batch = ((x, x_name), (x_cond, x_cond_name))
        
        # Call the parent sample method
        super().sample(net, new_batch, sample_path, stage)
