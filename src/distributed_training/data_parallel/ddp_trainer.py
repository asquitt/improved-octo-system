"""
Distributed Data Parallel (DDP) Trainer

This module implements PyTorch's DistributedDataParallel for efficient multi-GPU training.

Key Concepts:
-------------
1. **Data Parallelism**: Each GPU gets a complete copy of the model
2. **Gradient Synchronization**: After backward pass, gradients are averaged across all GPUs
3. **Process Groups**: NCCL backend for efficient GPU communication
4. **Rank**: Unique identifier for each process (0 to world_size-1)
5. **World Size**: Total number of processes (typically = number of GPUs)

How it works:
-------------
1. Initialize process group for inter-GPU communication
2. Wrap model with DDP for automatic gradient synchronization
3. Use DistributedSampler to ensure each GPU gets different data
4. Train normally - DDP handles the parallelism!

Author: Your Name
Date: 2025-11
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Callable, Dict, Any
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDPTrainer:
    """
    Distributed Data Parallel Trainer for multi-GPU training.

    This class handles all the complexity of setting up distributed training,
    including process initialization, model wrapping, and data distribution.

    Example:
    --------
    >>> model = MyModel()
    >>> trainer = DDPTrainer(model, num_gpus=4)
    >>> trainer.train(train_loader, num_epochs=10)

    Attributes:
    -----------
    model : nn.Module
        The neural network model to train
    rank : int
        Process rank (GPU ID) in the distributed setup
    world_size : int
        Total number of processes (GPUs)
    device : torch.device
        The GPU device this process uses
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        num_gpus: Optional[int] = None,
        backend: str = "nccl",
        checkpoint_dir: str = "./checkpoints",
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
    ):
        """
        Initialize the DDP Trainer.

        Parameters:
        -----------
        model : nn.Module
            The model to train
        optimizer : torch.optim.Optimizer, optional
            Optimizer for training. If None, uses Adam
        loss_fn : Callable, optional
            Loss function. If None, uses CrossEntropyLoss
        num_gpus : int, optional
            Number of GPUs to use. If None, uses all available
        backend : str
            Communication backend ('nccl' for GPU, 'gloo' for CPU)
        checkpoint_dir : str
            Directory to save checkpoints
        gradient_accumulation_steps : int
            Number of steps to accumulate gradients (for larger effective batch size)
        mixed_precision : bool
            Whether to use automatic mixed precision (AMP) training
        """
        # Store configuration
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = checkpoint_dir

        # Initialize distributed training
        self._init_distributed(backend, num_gpus)

        # Move model to GPU and wrap with DDP
        self.device = torch.device(f"cuda:{self.rank}")
        self.model = model.to(self.device)

        # Wrap model with DDP for automatic gradient synchronization
        # find_unused_parameters=True handles models with conditional paths
        self.model = DDP(
            self.model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False,  # Set to True if model has unused parameters
        )

        # Set up optimizer and loss function
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        # Set up automatic mixed precision if requested
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Create checkpoint directory
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(f"[Rank {self.rank}] DDPTrainer initialized")
        logger.info(f"[Rank {self.rank}] Device: {self.device}")
        logger.info(f"[Rank {self.rank}] World size: {self.world_size}")
        logger.info(f"[Rank {self.rank}] Mixed precision: {mixed_precision}")

    def _init_distributed(self, backend: str, num_gpus: Optional[int] = None):
        """
        Initialize the distributed training environment.

        This sets up the process group for inter-GPU communication.

        Parameters:
        -----------
        backend : str
            'nccl' for GPU training (recommended), 'gloo' for CPU
        num_gpus : int, optional
            Number of GPUs to use
        """
        # Check if running in distributed mode
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Running with torchrun or similar launcher
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # Single-node training - we'll set it up manually
            self.rank = 0
            self.world_size = num_gpus or torch.cuda.device_count()
            self.local_rank = 0

            # For single-process, multi-GPU, we need to spawn processes
            # For simplicity in this trainer, we assume torchrun is used
            logger.warning(
                "Not running with torchrun. For multi-GPU, use: "
                "torchrun --nproc_per_node=<num_gpus> your_script.py"
            )

        # Initialize process group
        if not dist.is_initialized():
            # NCCL is NVIDIA's optimized communication library for GPUs
            dist.init_process_group(
                backend=backend,
                init_method="env://",  # Use environment variables
                world_size=self.world_size,
                rank=self.rank,
            )

        # Set device for this process
        torch.cuda.set_device(self.local_rank)

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        log_interval: int = 10,
        save_interval: int = 1,
    ):
        """
        Train the model using distributed data parallel.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader (should use DistributedSampler)
        num_epochs : int
            Number of training epochs
        val_loader : DataLoader, optional
            Validation data loader
        log_interval : int
            How often to log training stats (in steps)
        save_interval : int
            How often to save checkpoints (in epochs)
        """
        logger.info(f"[Rank {self.rank}] Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Set epoch for DistributedSampler (ensures different shuffling each epoch)
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_loss = self._train_epoch(train_loader, epoch, log_interval)

            # Only main process logs and saves
            if self.is_main_process:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

                # Validation
                if val_loader is not None:
                    val_loss = self._validate(val_loader)
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")

                # Save checkpoint
                if (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(epoch, train_loss)

        logger.info(f"[Rank {self.rank}] Training complete!")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int,
    ) -> float:
        """
        Train for one epoch.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        epoch : int
            Current epoch number
        log_interval : int
            Logging frequency

        Returns:
        --------
        float : Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar only on main process
        iterator = enumerate(train_loader)
        if self.is_main_process:
            iterator = tqdm(iterator, total=len(train_loader), desc=f"Epoch {epoch+1}")

        for batch_idx, (data, target) in iterator:
            # Move data to GPU
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass with optional mixed precision
            if self.mixed_precision:
                # Automatic Mixed Precision (AMP) for faster training
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.loss_fn(output, target)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                # Standard training
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            # Gradient accumulation: only update weights every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    # Unscale gradients and step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Zero gradients for next accumulation
                self.optimizer.zero_grad()

            # Track loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Log progress
            if batch_idx % log_interval == 0 and self.is_main_process:
                current_loss = total_loss / num_batches
                logger.info(
                    f"[Rank {self.rank}] Epoch {epoch+1} "
                    f"[{batch_idx}/{len(train_loader)}] Loss: {current_loss:.4f}"
                )

        return total_loss / num_batches

    def _validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader

        Returns:
        --------
        float : Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, epoch: int, loss: float):
        """
        Save a training checkpoint.

        Only the main process saves checkpoints to avoid conflicts.

        Parameters:
        -----------
        epoch : int
            Current epoch number
        loss : float
            Current training loss
        """
        if not self.is_main_process:
            return

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch+1}.pt"
        )

        # Save model state (unwrap DDP first)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }

        if self.mixed_precision:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a training checkpoint.

        Parameters:
        -----------
        checkpoint_path : str
            Path to the checkpoint file
        """
        logger.info(f"[Rank {self.rank}] Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.mixed_precision and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"[Rank {self.rank}] Checkpoint loaded successfully")
        return checkpoint["epoch"], checkpoint["loss"]

    def cleanup(self):
        """Clean up distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info(f"[Rank {self.rank}] Cleanup complete")


def setup_distributed_sampler(dataset, world_size: int, rank: int, shuffle: bool = True):
    """
    Helper function to create a DistributedSampler.

    DistributedSampler ensures each GPU gets different data batches.

    Parameters:
    -----------
    dataset : torch.utils.data.Dataset
        The dataset to sample from
    world_size : int
        Total number of processes
    rank : int
        Current process rank
    shuffle : bool
        Whether to shuffle the data

    Returns:
    --------
    DistributedSampler
    """
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )
