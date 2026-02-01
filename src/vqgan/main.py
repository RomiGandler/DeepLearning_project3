"""
VQGAN CLI Entry Point - Command-line interface for training and testing.

Usage:
    Training from scratch:
        python -m src.vqgan.main --config src/vqgan/config.yaml --dataset /path/to/data
    
    Finetune from HuggingFace:
        python -m src.vqgan.main --config src/vqgan/config.yaml --dataset /path/to/data
        (with ckpt: "vqgan_f8.ckpt" in config)
    
    Resume training:
        python -m src.vqgan.main --config src/vqgan/config.yaml --resume logs/vqgan/.../last.ckpt
    
    Test only:
        python -m src.vqgan.main --config src/vqgan/config.yaml --resume checkpoints/model.ckpt --test-only
"""
import argparse
import sys
import os
from pytorch_lightning import seed_everything


from src.vqgan.trainer import VQGANTrainer



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VQGAN Training and Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required/common arguments
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config file (default: src/vqgan/config_train.yaml)"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Dataset path (overrides config). If not provided, downloads from HuggingFace."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Path to checkpoint to train and test"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="logs/vqgan",
        help="Output directory for logs and checkpoints (default: logs/vqgan)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use, 0 for CPU (default: 1)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=[16, 32],
        default=32,
        help="Training precision: 16 or 32 (default: 32)"
    )
    parser.add_argument(
        "--accumulate-grad",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    
    # Mode selection
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run testing only (requires --resume)"
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if args.test_only and args.model is None:
        print("Warning: --test-only requires --model to specify the model to test.")
        print("Will attempt to use checkpoint from config if specified.")
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Resolve config path relative to the script if not found as is
    config_path = args.config
    if not os.path.exists(config_path):
        # Try finding it relative to project root or current working directory
        potential_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), config_path)
        if os.path.exists(potential_path):
            config_path = potential_path
        elif not os.path.isabs(config_path):
             # Try absolute path based on cwd
             potential_path = os.path.abspath(config_path)
             if os.path.exists(potential_path):
                 config_path = potential_path

    # Create trainer
    trainer = VQGANTrainer(
        config_path=config_path,
        dataset_path=args.dataset,
        checkpoint_path=args.model,
        output_dir=args.output,
    )
    
    # Run training or testing
    test_only = args.test_only
    if not test_only and "training" in trainer.config:
        test_only = trainer.config.training.get("test_only", False)

    if test_only:
        trainer.test()
    else:
        trainer.train(
            max_epochs=args.epochs,
            gpus=args.gpus,
            precision=args.precision,
            accumulate_grad_batches=args.accumulate_grad,
        )


if __name__ == "__main__":
    main()
