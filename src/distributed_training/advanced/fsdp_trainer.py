"""
Fully Sharded Data Parallel (FSDP) Trainer

FSDP is PyTorch's native implementation of ZeRO-style sharding.
It shards model parameters, gradients, and optimizer states across GPUs.

Benefits over DDP:
------------------
- Reduced memory per GPU (shard parameters)
- Train larger models
- Native PyTorch integration (no external dependencies)

When to use FSDP vs DeepSpeed:
------------------------------
- FSDP: Native PyTorch, simpler, good for most cases
- DeepSpeed: More features, CPU offloading, better for extreme scale

Example:
--------
>>> trainer = FSDPTrainer(
...     model=model,
...     sharding_strategy="FULL_SHARD",
...     mixed_precision=True
... )
>>> trainer.train(train_loader, num_epochs=10)

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.utils.data import DataLoader
from typing import Optional, Callable
from functools import partial
import logging

logger = logging.getLogger(__name__)


class FSDPTrainer:
    """
    Trainer using Fully Sharded Data Parallel.

    FSDP shards model parameters, gradients, and optimizer states
    across data parallel workers, reducing memory per GPU.

    Parameters:
    -----------
    model : nn.Module
        Model to train
    sharding_strategy : str
        Sharding strategy: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
    mixed_precision : bool
        Enable mixed precision training
    cpu_offload : bool
        Offload parameters to CPU
    auto_wrap_policy : str
        Auto-wrap policy: "size_based" or "transformer"
    min_num_params : int
        Minimum parameters for wrapping (size_based policy)
    """

    def __init__(
        self,
        model: nn.Module,
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision: bool = True,
        cpu_offload: bool = False,
        auto_wrap_policy: str = "size_based",
        min_num_params: int = 1e6,
    ):
        """Initialize FSDP trainer."""
        self.sharding_strategy = self._get_sharding_strategy(sharding_strategy)
        self.mixed_precision = mixed_precision
        self.cpu_offload = cpu_offload
        self.auto_wrap_policy = auto_wrap_policy
        self.min_num_params = int(min_num_params)

        # Wrap model with FSDP
        self.model = self._wrap_model(model)

        logger.info(f"FSDPTrainer initialized with {sharding_strategy} sharding")

    def _get_sharding_strategy(self, strategy_name: str) -> ShardingStrategy:
        """Convert string to ShardingStrategy enum."""
        strategies = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }
        return strategies.get(strategy_name, ShardingStrategy.FULL_SHARD)

    def _get_mixed_precision_policy(self) -> Optional[MixedPrecision]:
        """Get mixed precision configuration."""
        if not self.mixed_precision:
            return None

        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    def _get_auto_wrap_policy(self):
        """Get auto-wrap policy for FSDP."""
        if self.auto_wrap_policy == "size_based":
            return partial(
                size_based_auto_wrap_policy,
                min_num_params=self.min_num_params
            )
        elif self.auto_wrap_policy == "transformer":
            # Wrap transformer blocks
            return partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    nn.TransformerEncoderLayer,
                    nn.TransformerDecoderLayer,
                }
            )
        return None

    def _wrap_model(self, model: nn.Module) -> FSDP:
        """Wrap model with FSDP."""
        # CPU offload configuration
        cpu_offload_config = None
        if self.cpu_offload:
            cpu_offload_config = CPUOffload(offload_params=True)

        # Wrap with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=self.sharding_strategy,
            mixed_precision=self._get_mixed_precision_policy(),
            cpu_offload=cpu_offload_config,
            auto_wrap_policy=self._get_auto_wrap_policy(),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
        )

        logger.info("Model wrapped with FSDP")
        return fsdp_model

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        val_loader: Optional[DataLoader] = None,
        log_interval: int = 10,
    ):
        """
        Train with FSDP.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        num_epochs : int
            Number of epochs
        optimizer : torch.optim.Optimizer
            Optimizer
        loss_fn : Callable
            Loss function
        val_loader : DataLoader, optional
            Validation loader
        log_interval : int
            Logging frequency
        """
        logger.info(f"Starting FSDP training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.cuda()
                target = target.cuda()

                # Forward pass
                output = self.model(data)
                loss = loss_fn(output, target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % log_interval == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} "
                        f"[{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

            # Validation
            if val_loader:
                val_loss = self._validate(val_loader, loss_fn)
                logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")

    def _validate(self, val_loader: DataLoader, loss_fn: Callable) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.cuda()
                target = target.cuda()

                output = self.model(data)
                loss = loss_fn(output, target)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, filepath: str):
        """Save FSDP checkpoint."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = self.model.state_dict()

            if torch.distributed.get_rank() == 0:
                torch.save(state_dict, filepath)
                logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load FSDP checkpoint."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )

        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, load_policy):
            state_dict = torch.load(filepath)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint from {filepath}")
