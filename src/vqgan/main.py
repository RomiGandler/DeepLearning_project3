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

from pytorch_lightning import seed_everything

from .trainer import VQGANTrainer


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
        default="src/vqgan/config.yaml",
        help="Path to config file (default: src/vqgan/config.yaml)"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Dataset path (overrides config). If not provided, downloads from HuggingFace."
    )
    parser.add_argument(
        "-r", "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (or for testing)"
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
    if args.test_only and args.resume is None:
        print("Warning: --test-only requires --resume to specify the checkpoint to test.")
        print("Will attempt to use checkpoint from config if specified.")
    
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Create trainer
    trainer = VQGANTrainer(
        config_path=args.config,
        dataset_path=args.dataset,
        checkpoint_path=args.resume,
        output_dir=args.output,
    )
    
    # Run training or testing
    if args.test_only:
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
