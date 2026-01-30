"""
BBDM Training and Inference Entry Point.

Usage:
    python -m src.bbdm.main -c configs/mlbbdm-f8-075.yaml -t  # Train
    python -m src.bbdm.main -c configs/mlbbdm-f8-075.yaml --sample_to_eval  # Evaluate
"""
import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np

from src.bbdm.utils import dict2namespace, get_runner, namespace2dict
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='BBDM Training and Inference')

    parser.add_argument('-c', '--config', type=str, required=True, 
                        help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, 
                        help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', 
                        help="The directory to save results")

    parser.add_argument('-t', '--train', action='store_true', default=False, 
                        help='Train the model')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, 
                        help='Sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, 
                        help='Sample at start (for debug)')
    parser.add_argument('--save_top', action='store_true', default=False, 
                        help="Save top loss checkpoint")

    parser.add_argument('--gpu_ids', type=str, default='0', 
                        help='GPU ids, e.g. 0,1,2,3. Use -1 for CPU')
    parser.add_argument('--port', type=str, default='12355', 
                        help='DDP master port')

    parser.add_argument('--resume_model', type=str, default=None, 
                        help='Model checkpoint path to resume from')
    parser.add_argument('--resume_optim', type=str, default=None, 
                        help='Optimizer checkpoint path to resume from')

    parser.add_argument('--max_epoch', type=int, default=None, 
                        help='Maximum number of epochs')
    parser.add_argument('--max_steps', type=int, default=None, 
                        help='Maximum number of steps')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    # Override config with command line arguments
    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    # ------------------------------------------------------------------
    # Validate required checkpoints (explicit user choice)
    #
    # Project policy:
    # - User MUST specify both VQGAN and BBDM checkpoints for training/test.
    # - Values may be either:
    #   - a filename (resolved under ./checkpoints and downloaded from HF if missing)
    #   - an absolute/relative filesystem path
    # ------------------------------------------------------------------
    if not hasattr(namespace_config, "model"):
        raise ValueError("Config missing required section: model")
    if not hasattr(namespace_config.model, "VQGAN") or not hasattr(namespace_config.model.VQGAN, "params"):
        raise ValueError("Config missing required section: model.VQGAN.params")

    vqgan_ckpt = getattr(namespace_config.model.VQGAN.params, "ckpt_path", None)
    if vqgan_ckpt is None or str(vqgan_ckpt).strip() == "":
        raise ValueError(
            "You must set `model.VQGAN.params.ckpt_path` in the config (filename or path). "
            "It will be downloaded into `./checkpoints/` if missing."
        )

    bbdm_ckpt = getattr(namespace_config.model, "model_load_path", None)
    # Only require BBDM checkpoint for inference, not for training from scratch
    if not args.train and (bbdm_ckpt is None or str(bbdm_ckpt).strip() == ""):
        raise ValueError(
            "You must set `model.model_load_path` in the config for inference. "
            "It will be downloaded into `./checkpoints/` if missing."
        )

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def DDP_run_fn(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    set_random_seed(config.args.seed)

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    config.training.device = [torch.device("cuda:%d" % local_rank)]
    print('using device:', config.training.device)
    config.training.local_rank = local_rank
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def CPU_singleGPU_launcher(config):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    if config.args.train:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def DDP_launcher(world_size, run_fn, config):
    mp.spawn(run_fn,
             args=(world_size, copy.deepcopy(config)),
             nprocs=world_size,
             join=True)


def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args

    gpu_ids = args.gpu_ids
    if gpu_ids == "-1":  # Use CPU
        nconfig.training.use_DDP = False
        nconfig.training.device = [torch.device("cpu")]
        CPU_singleGPU_launcher(nconfig)
    else:
        gpu_list = gpu_ids.split(",")
        if len(gpu_list) > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            nconfig.training.use_DDP = True
            DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, config=nconfig)
        else:
            nconfig.training.use_DDP = False
            nconfig.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
            CPU_singleGPU_launcher(nconfig)
    return


if __name__ == "__main__":
    main()
