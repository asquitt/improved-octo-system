"""
Training Configuration Management

Centralized configuration for all training parameters.
Supports YAML files for easy experimentation.

Author: Your Name
Date: 2025-11
"""

import yaml
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Complete training configuration.

    All training parameters in one place for easy management
    and reproducibility.

    Example:
    --------
    >>> config = TrainingConfig(
    ...     num_epochs=10,
    ...     batch_size=32,
    ...     learning_rate=0.001
    ... )
    >>> config.save("config.yaml")
    """

    # Training
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1

    # Distributed
    strategy: str = "ddp"  # ddp, fsdp, deepspeed, model_parallel
    num_gpus: int = 1
    num_nodes: int = 1

    # DeepSpeed (if using)
    deepspeed_zero_stage: int = 2
    deepspeed_offload_optimizer: bool = False
    deepspeed_offload_param: bool = False

    # Optimization
    optimizer: str = "adam"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, linear, step, constant
    warmup_steps: int = 0
    max_grad_norm: float = 1.0

    # Mixed Precision
    mixed_precision: bool = True
    fp16: bool = True
    bf16: bool = False

    # Data Loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1
    max_checkpoints: int = 3
    save_best: bool = True
    best_metric: str = "loss"
    best_mode: str = "min"

    # Logging
    log_interval: int = 10
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Model
    model_name: str = "resnet50"
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Dataset
    dataset_name: str = "cifar10"
    data_dir: str = "./data"
    train_split: str = "train"
    val_split: str = "val"

    # Advanced
    activation_checkpointing: bool = False
    gradient_compression: bool = False
    find_unused_parameters: bool = False

    # Reproducibility
    seed: int = 42
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True

    def save(self, filepath: str):
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        logger.info(f"Loaded config from {filepath}")
        return cls(**config_dict)

    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []

        # Check positive values
        if self.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")

        # Check strategy
        valid_strategies = ["ddp", "fsdp", "deepspeed", "model_parallel", "pipeline_parallel"]
        if self.strategy not in valid_strategies:
            errors.append(f"strategy must be one of {valid_strategies}")

        # Check optimizer
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop"]
        if self.optimizer not in valid_optimizers:
            errors.append(f"optimizer must be one of {valid_optimizers}")

        # Check scheduler
        valid_schedulers = ["cosine", "linear", "step", "constant", "exponential"]
        if self.scheduler not in valid_schedulers:
            errors.append(f"scheduler must be one of {valid_schedulers}")

        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False

        logger.info("Configuration validated successfully")
        return True

    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("Training Configuration Summary")
        print("="*60)

        print(f"\n📊 Training:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Scheduler: {self.scheduler}")

        print(f"\n🚀 Distributed:")
        print(f"  Strategy: {self.strategy}")
        print(f"  GPUs: {self.num_gpus}")
        print(f"  Nodes: {self.num_nodes}")

        print(f"\n💾 Checkpointing:")
        print(f"  Directory: {self.checkpoint_dir}")
        print(f"  Interval: {self.checkpoint_interval}")
        print(f"  Save Best: {self.save_best}")

        print(f"\n📈 Logging:")
        print(f"  TensorBoard: {self.use_tensorboard}")
        print(f"  Weights & Biases: {self.use_wandb}")

        print("="*60 + "\n")


def load_config(filepath: str) -> TrainingConfig:
    """
    Load training configuration from file.

    Parameters:
    -----------
    filepath : str
        Path to YAML config file

    Returns:
    --------
    TrainingConfig : Loaded configuration
    """
    return TrainingConfig.load(filepath)


def save_config(config: TrainingConfig, filepath: str):
    """
    Save training configuration to file.

    Parameters:
    -----------
    config : TrainingConfig
        Configuration to save
    filepath : str
        Output path
    """
    config.save(filepath)


def create_default_configs():
    """
    Create default configuration files for different scenarios.

    Creates configs for:
    - Single GPU training
    - Multi-GPU DDP training
    - DeepSpeed training
    - FSDP training
    """
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Single GPU
    single_gpu = TrainingConfig(
        num_epochs=10,
        batch_size=32,
        strategy="ddp",
        num_gpus=1,
    )
    single_gpu.save("configs/single_gpu.yaml")

    # Multi-GPU DDP
    multi_gpu = TrainingConfig(
        num_epochs=10,
        batch_size=128,  # Total across GPUs
        strategy="ddp",
        num_gpus=4,
        mixed_precision=True,
    )
    multi_gpu.save("configs/multi_gpu_ddp.yaml")

    # DeepSpeed ZeRO-2
    deepspeed = TrainingConfig(
        num_epochs=10,
        batch_size=128,
        strategy="deepspeed",
        num_gpus=4,
        deepspeed_zero_stage=2,
        deepspeed_offload_optimizer=False,
    )
    deepspeed.save("configs/deepspeed_zero2.yaml")

    # DeepSpeed ZeRO-3 with offloading
    deepspeed_offload = TrainingConfig(
        num_epochs=10,
        batch_size=128,
        strategy="deepspeed",
        num_gpus=4,
        deepspeed_zero_stage=3,
        deepspeed_offload_optimizer=True,
        deepspeed_offload_param=True,
    )
    deepspeed_offload.save("configs/deepspeed_zero3_offload.yaml")

    # FSDP
    fsdp = TrainingConfig(
        num_epochs=10,
        batch_size=128,
        strategy="fsdp",
        num_gpus=4,
        mixed_precision=True,
        activation_checkpointing=True,
    )
    fsdp.save("configs/fsdp.yaml")

    logger.info("Created default configuration files in configs/")
