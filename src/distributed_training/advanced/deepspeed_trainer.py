"""
DeepSpeed ZeRO Integration

DeepSpeed ZeRO (Zero Redundancy Optimizer) is a memory optimization technique
that partitions optimizer states, gradients, and parameters across GPUs.

ZeRO Stages:
------------
- **Stage 1**: Partition optimizer states → 4x memory reduction
- **Stage 2**: Partition optimizer states + gradients → 8x memory reduction
- **Stage 3**: Partition optimizer states + gradients + parameters → Linear with #GPUs

Benefits:
---------
- Train models 10x-100x larger than standard DDP
- Better memory efficiency than model parallelism
- Near-linear scaling with more GPUs

Example:
--------
>>> trainer = DeepSpeedTrainer(
...     model=large_model,
...     zero_stage=2,  # Use ZeRO stage 2
...     config_file="deepspeed_config.json"
... )
>>> trainer.train(train_loader, num_epochs=10)

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging
import json
import os

logger = logging.getLogger(__name__)


class DeepSpeedTrainer:
    """
    Trainer with DeepSpeed ZeRO optimization.

    This trainer integrates Microsoft's DeepSpeed library for
    memory-efficient training of very large models.

    Key Features:
    -------------
    - ZeRO optimizer state partitioning
    - Gradient and parameter partitioning
    - Automatic mixed precision (FP16/BF16)
    - CPU offloading for even larger models
    - Gradient accumulation
    - Dynamic loss scaling

    Parameters:
    -----------
    model : nn.Module
        Model to train
    zero_stage : int
        ZeRO optimization stage (1, 2, or 3)
    config_file : str, optional
        Path to DeepSpeed config JSON
    offload_optimizer : bool
        Offload optimizer states to CPU
    offload_param : bool
        Offload parameters to CPU (ZeRO-3 only)
    """

    def __init__(
        self,
        model: nn.Module,
        zero_stage: int = 2,
        config_file: Optional[str] = None,
        offload_optimizer: bool = False,
        offload_param: bool = False,
        micro_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
    ):
        """Initialize DeepSpeed trainer."""
        self.model = model
        self.zero_stage = zero_stage
        self.config_file = config_file
        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Generate DeepSpeed config
        if config_file is None:
            self.ds_config = self._generate_deepspeed_config()
        else:
            with open(config_file, 'r') as f:
                self.ds_config = json.load(f)

        logger.info(f"DeepSpeedTrainer initialized with ZeRO stage {zero_stage}")

    def _generate_deepspeed_config(self) -> Dict[str, Any]:
        """
        Generate DeepSpeed configuration automatically.

        Returns:
        --------
        dict : DeepSpeed configuration
        """
        config = {
            "train_batch_size": self.micro_batch_size * self.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "steps_per_print": 10,
            "wall_clock_breakdown": False,

            # Mixed precision training
            "fp16": {
                "enabled": True,
                "loss_scale": 0,  # Dynamic loss scaling
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },

            # ZeRO optimization
            "zero_optimization": {
                "stage": self.zero_stage,
            },

            # Gradient clipping
            "gradient_clipping": 1.0,
        }

        # Stage-specific configurations
        if self.zero_stage >= 2:
            config["zero_optimization"]["contiguous_gradients"] = True
            config["zero_optimization"]["overlap_comm"] = True

        if self.zero_stage == 3:
            config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e8
            config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6

            # CPU offloading (optional)
            if self.offload_param:
                config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

        # Optimizer offloading (works with all stages)
        if self.offload_optimizer:
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        logger.info("Generated DeepSpeed configuration")
        logger.debug(f"Config: {json.dumps(config, indent=2)}")

        return config

    def initialize(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        """
        Initialize DeepSpeed engine.

        Parameters:
        -----------
        model : nn.Module
            Model to train
        optimizer : torch.optim.Optimizer, optional
            Optimizer (if None, uses Adam)
        lr_scheduler : optional
            Learning rate scheduler

        Returns:
        --------
        tuple : (model_engine, optimizer, _, lr_scheduler)
        """
        try:
            import deepspeed
        except ImportError:
            raise ImportError(
                "DeepSpeed not installed. Install with: pip install deepspeed"
            )

        # Initialize DeepSpeed
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=self.ds_config,
        )

        self.model_engine = model_engine
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        logger.info("DeepSpeed engine initialized")

        return model_engine, optimizer, _, lr_scheduler

    def train_step(
        self,
        batch: tuple,
        loss_fn: callable,
    ) -> float:
        """
        Perform one training step.

        Parameters:
        -----------
        batch : tuple
            (data, target) batch
        loss_fn : callable
            Loss function

        Returns:
        --------
        float : Loss value
        """
        data, target = batch

        # Forward pass
        output = self.model_engine(data)
        loss = loss_fn(output, target)

        # Backward pass (DeepSpeed handles gradient accumulation)
        self.model_engine.backward(loss)

        # Optimizer step (DeepSpeed handles when to step)
        self.model_engine.step()

        return loss.item()

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        loss_fn: callable,
        val_loader: Optional[DataLoader] = None,
        log_interval: int = 10,
    ):
        """
        Train with DeepSpeed.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        num_epochs : int
            Number of epochs
        loss_fn : callable
            Loss function
        val_loader : DataLoader, optional
            Validation data loader
        log_interval : int
            Logging frequency
        """
        logger.info(f"Starting DeepSpeed training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.model_engine.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                loss = self.train_step(batch, loss_fn)

                total_loss += loss
                num_batches += 1

                if batch_idx % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} "
                        f"[{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss:.4f}"
                    )

            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader, loss_fn)
                logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")

    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: callable,
    ) -> float:
        """
        Validate model.

        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader
        loss_fn : callable
            Loss function

        Returns:
        --------
        float : Average validation loss
        """
        self.model_engine.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                data, target = batch
                output = self.model_engine(data)
                loss = loss_fn(output, target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, save_dir: str, tag: str = "latest"):
        """
        Save DeepSpeed checkpoint.

        Parameters:
        -----------
        save_dir : str
            Directory to save checkpoint
        tag : str
            Checkpoint tag
        """
        self.model_engine.save_checkpoint(save_dir, tag=tag)
        logger.info(f"Saved checkpoint to {save_dir}/{tag}")

    def load_checkpoint(
        self,
        load_dir: str,
        tag: str = "latest",
    ):
        """
        Load DeepSpeed checkpoint.

        Parameters:
        -----------
        load_dir : str
            Directory containing checkpoint
        tag : str
            Checkpoint tag
        """
        _, client_state = self.model_engine.load_checkpoint(load_dir, tag=tag)
        logger.info(f"Loaded checkpoint from {load_dir}/{tag}")
        return client_state


def create_deepspeed_config(
    zero_stage: int = 2,
    train_batch_size: int = 32,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: Optional[int] = None,
    fp16_enabled: bool = True,
    bf16_enabled: bool = False,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    output_file: str = "deepspeed_config.json",
) -> Dict[str, Any]:
    """
    Create a DeepSpeed configuration file.

    This helper function generates a complete DeepSpeed config
    with recommended settings.

    Parameters:
    -----------
    zero_stage : int
        ZeRO optimization stage (1, 2, or 3)
    train_batch_size : int
        Total batch size across all GPUs
    micro_batch_size : int
        Batch size per GPU per step
    gradient_accumulation_steps : int, optional
        Steps to accumulate gradients
    fp16_enabled : bool
        Enable FP16 mixed precision
    bf16_enabled : bool
        Enable BF16 mixed precision (A100+ GPUs)
    offload_optimizer : bool
        Offload optimizer to CPU
    offload_param : bool
        Offload parameters to CPU (ZeRO-3 only)
    output_file : str
        Output configuration file path

    Returns:
    --------
    dict : DeepSpeed configuration

    Example:
    --------
    >>> config = create_deepspeed_config(
    ...     zero_stage=3,
    ...     train_batch_size=128,
    ...     micro_batch_size=4,
    ...     offload_optimizer=True
    ... )
    >>> # Config saved to deepspeed_config.json
    """
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = train_batch_size // micro_batch_size

    config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": True,

        "zero_optimization": {
            "stage": zero_stage,
        },
    }

    # Mixed precision
    if fp16_enabled:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    elif bf16_enabled:
        config["bf16"] = {"enabled": True}

    # ZeRO stage-specific settings
    if zero_stage >= 2:
        config["zero_optimization"]["contiguous_gradients"] = True
        config["zero_optimization"]["overlap_comm"] = True
        config["zero_optimization"]["allgather_bucket_size"] = 5e8
        config["zero_optimization"]["reduce_bucket_size"] = 5e8

    if zero_stage == 3:
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e8
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6
        config["zero_optimization"]["stage3_max_live_parameters"] = 1e9
        config["zero_optimization"]["stage3_max_reuse_distance"] = 1e9
        config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

    # CPU offloading
    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if offload_param and zero_stage == 3:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"DeepSpeed config saved to {output_file}")
    logger.info(f"Configuration summary:")
    logger.info(f"  ZeRO Stage: {zero_stage}")
    logger.info(f"  Batch Size: {train_batch_size}")
    logger.info(f"  Micro Batch Size: {micro_batch_size}")
    logger.info(f"  Gradient Accumulation: {gradient_accumulation_steps}")
    logger.info(f"  FP16: {fp16_enabled}")
    logger.info(f"  Offload Optimizer: {offload_optimizer}")
    logger.info(f"  Offload Param: {offload_param}")

    return config
