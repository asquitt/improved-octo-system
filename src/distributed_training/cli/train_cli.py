#!/usr/bin/env python3
"""
Command-Line Interface for Distributed Training

Provides easy-to-use CLI for running distributed training with
various strategies and configurations.

Usage:
------
# Basic training
python -m distributed_training.cli.train_cli --config configs/ddp.yaml

# With custom parameters
python -m distributed_training.cli.train_cli \
    --config configs/ddp.yaml \
    --num-epochs 20 \
    --batch-size 64

# Find optimal learning rate
python -m distributed_training.cli.train_cli \
    --config configs/ddp.yaml \
    --find-lr

Author: Your Name
Date: 2025-11
"""

import argparse
import sys
sys.path.insert(0, 'src')

from distributed_training.config import TrainingConfig, create_default_configs
from distributed_training.optimization import find_optimal_batch_size, LRFinder
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  python -m distributed_training.cli.train_cli --config configs/ddp.yaml

  # Find optimal batch size
  python -m distributed_training.cli.train_cli --find-batch-size

  # Find optimal learning rate
  python -m distributed_training.cli.train_cli --config configs/ddp.yaml --find-lr

  # Create default configs
  python -m distributed_training.cli.train_cli --create-configs
        """
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration YAML file"
    )

    # Utilities
    parser.add_argument(
        "--create-configs",
        action="store_true",
        help="Create default configuration files"
    )

    parser.add_argument(
        "--find-batch-size",
        action="store_true",
        help="Find optimal batch size for model"
    )

    parser.add_argument(
        "--find-lr",
        action="store_true",
        help="Find optimal learning rate"
    )

    # Override parameters
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )

    parser.add_argument(
        "--strategy",
        choices=["ddp", "fsdp", "deepspeed", "model_parallel"],
        help="Distributed training strategy"
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs to use"
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Data directory"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Checkpoint directory"
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Create default configs if requested
    if args.create_configs:
        logger.info("Creating default configuration files...")
        create_default_configs()
        logger.info("✓ Default configs created in configs/")
        return

    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = TrainingConfig.load(args.config)
    else:
        logger.info("Using default configuration")
        config = TrainingConfig()

    # Override with command-line args
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.strategy:
        config.strategy = args.strategy
    if args.num_gpus:
        config.num_gpus = args.num_gpus
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    # Validate config
    if not config.validate():
        logger.error("Configuration validation failed!")
        sys.exit(1)

    # Print summary
    config.print_summary()

    # Find optimal batch size if requested
    if args.find_batch_size:
        logger.info("Finding optimal batch size...")
        logger.info("This feature requires model implementation")
        logger.info("Implement your model and use:")
        logger.info("  from distributed_training.optimization import find_optimal_batch_size")
        logger.info("  optimal_bs = find_optimal_batch_size(model, input_shape=(3, 224, 224))")
        return

    # Find optimal learning rate if requested
    if args.find_lr:
        logger.info("Finding optimal learning rate...")
        logger.info("This feature requires model and data loader")
        logger.info("Implement your model and use:")
        logger.info("  from distributed_training.optimization import LRFinder")
        logger.info("  lr_finder = LRFinder(model, optimizer, criterion)")
        logger.info("  lr_finder.range_test(train_loader)")
        logger.info("  lr_finder.plot()")
        return

    # Start training
    logger.info("Starting training...")
    logger.info(f"Strategy: {config.strategy}")
    logger.info(f"GPUs: {config.num_gpus}")
    logger.info(f"Epochs: {config.num_epochs}")

    logger.info("\nTo implement training, create your model and data loaders,")
    logger.info("then use the appropriate trainer:")
    logger.info("")
    logger.info("from distributed_training.data_parallel import DDPTrainer")
    logger.info("from distributed_training.advanced import FSDPTrainer, DeepSpeedTrainer")
    logger.info("")
    logger.info("trainer = DDPTrainer(model=model, ...)")
    logger.info("trainer.train(train_loader, num_epochs=config.num_epochs)")


if __name__ == "__main__":
    main()
