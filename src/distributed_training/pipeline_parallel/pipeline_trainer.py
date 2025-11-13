"""
Pipeline Parallelism Implementation

Pipeline parallelism partitions the model **by layers** across devices and
processes multiple micro-batches simultaneously for better efficiency.

Key Concepts:
-------------
1. **Pipeline Stages**: Partition model into sequential stages (each on different GPU)
2. **Micro-batches**: Split batch into smaller chunks for pipelining
3. **Scheduling**: Overlap forward and backward passes across stages
4. **Bubble Time**: Idle time at start/end of pipeline (minimize this!)

Example Pipeline with 4 Stages:
-------------------------------
Time →
Stage 0: [F1]──[F2]──[F3]──[F4]──[B1]──[B2]──[B3]──[B4]
Stage 1: ─────[F1]──[F2]──[F3]──[F4]──[B1]──[B2]──[B3]──[B4]
Stage 2: ────────[F1]──[F2]──[F3]──[F4]──[B1]──[B2]──[B3]──[B4]
Stage 3: ───────────[F1]──[F2]──[F3]──[F4]──[B1]──[B2]──[B3]──[B4]

F = Forward pass, B = Backward pass
Numbers = Micro-batch ID

Benefits:
---------
- Better efficiency than simple model parallelism
- Reduced bubble time with more micro-batches
- Memory efficient (only store one micro-batch per stage)

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import List, Optional, Callable, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PipelineStage:
    """
    Represents one stage in the pipeline.

    Each stage contains a portion of the model layers and runs on a separate GPU.

    Attributes:
    -----------
    stage_id : int
        This stage's ID (0 to num_stages-1)
    module : nn.Module
        The model layers for this stage
    device : torch.device
        GPU device for this stage
    """

    def __init__(
        self,
        stage_id: int,
        module: nn.Module,
        device: torch.device,
    ):
        """
        Initialize pipeline stage.

        Parameters:
        -----------
        stage_id : int
            Stage identifier
        module : nn.Module
            Model layers for this stage
        device : torch.device
            GPU device
        """
        self.stage_id = stage_id
        self.module = module.to(device)
        self.device = device

        logger.info(f"Pipeline Stage {stage_id} initialized on {device}")

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for this stage.

        Parameters:
        -----------
        input_tensor : torch.Tensor
            Input from previous stage (or input data for first stage)

        Returns:
        --------
        torch.Tensor : Output to next stage
        """
        input_tensor = input_tensor.to(self.device)
        return self.module(input_tensor)


class PipelineTrainer:
    """
    Pipeline Parallel Trainer using GPipe-style micro-batch scheduling.

    This trainer splits the model into stages and the batch into micro-batches,
    then pipelines execution for better GPU utilization.

    Key Features:
    -------------
    - Automatic model partitioning across GPUs
    - Micro-batch scheduling with minimal bubble time
    - Gradient accumulation across micro-batches
    - Support for custom partition strategies

    Example:
    --------
    >>> model = MyLargeModel()
    >>> trainer = PipelineTrainer(
    ...     model=model,
    ...     num_stages=4,
    ...     micro_batch_size=8
    ... )
    >>> trainer.train(train_loader, num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        num_stages: int,
        micro_batch_size: int,
        loss_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        partition_layers: Optional[List[int]] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Initialize pipeline parallel trainer.

        Parameters:
        -----------
        model : nn.Module
            The model to train (will be partitioned)
        num_stages : int
            Number of pipeline stages (= number of GPUs)
        micro_batch_size : int
            Size of each micro-batch
        loss_fn : Callable, optional
            Loss function
        optimizer : torch.optim.Optimizer, optional
            Optimizer for training
        partition_layers : List[int], optional
            Manual layer partition indices. If None, automatically balanced.
        checkpoint_dir : str
            Directory for saving checkpoints
        """
        self.num_stages = num_stages
        self.micro_batch_size = micro_batch_size
        self.checkpoint_dir = checkpoint_dir

        # Get rank and world size
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            logger.warning(
                "Distributed not initialized. For multi-GPU: "
                "torchrun --nproc_per_node=<num_stages> your_script.py"
            )

        assert self.world_size == num_stages, (
            f"World size ({self.world_size}) must equal num_stages ({num_stages})"
        )

        # Partition model into stages
        self.stages = self._partition_model(model, partition_layers)
        self.stage = self.stages[self.rank]  # This GPU's stage

        # Set up loss and optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(
            self.stage.module.parameters(), lr=1e-3
        )

        # Track activations and gradients for micro-batches
        self.micro_batch_activations = deque()
        self.micro_batch_gradients = deque()

        logger.info(
            f"[Stage {self.rank}] PipelineTrainer initialized with "
            f"{num_stages} stages, micro_batch_size={micro_batch_size}"
        )

    def _partition_model(
        self,
        model: nn.Module,
        partition_layers: Optional[List[int]] = None,
    ) -> List[PipelineStage]:
        """
        Partition model into pipeline stages.

        Parameters:
        -----------
        model : nn.Module
            Model to partition
        partition_layers : List[int], optional
            Indices where to partition. If None, split evenly.

        Returns:
        --------
        List[PipelineStage] : List of pipeline stages
        """
        # Get all layers (assumes model is nn.Sequential or has 'layers' attribute)
        if isinstance(model, nn.Sequential):
            layers = list(model)
        elif hasattr(model, "layers"):
            layers = list(model.layers)
        else:
            raise ValueError(
                "Model must be nn.Sequential or have 'layers' attribute. "
                "For custom models, wrap in nn.Sequential or implement "
                "custom partitioning."
            )

        total_layers = len(layers)

        # Determine partition boundaries
        if partition_layers is None:
            # Automatic balanced partitioning
            layers_per_stage = total_layers // self.num_stages
            partition_layers = [
                i * layers_per_stage for i in range(self.num_stages)
            ]
            partition_layers.append(total_layers)
        else:
            # Validate manual partition
            assert len(partition_layers) == self.num_stages + 1, (
                f"partition_layers must have {self.num_stages + 1} elements"
            )
            partition_layers = [0] + partition_layers + [total_layers]

        logger.info(f"Model partition boundaries: {partition_layers}")

        # Create stages
        stages = []
        for stage_id in range(self.num_stages):
            start_idx = partition_layers[stage_id]
            end_idx = partition_layers[stage_id + 1]

            stage_layers = layers[start_idx:end_idx]
            stage_module = nn.Sequential(*stage_layers)

            device = torch.device(f"cuda:{stage_id}")
            stage = PipelineStage(stage_id, stage_module, device)
            stages.append(stage)

            logger.debug(
                f"Stage {stage_id}: layers {start_idx}-{end_idx} "
                f"({len(stage_layers)} layers)"
            )

        return stages

    @property
    def is_first_stage(self) -> bool:
        """Check if this is the first pipeline stage."""
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        """Check if this is the last pipeline stage."""
        return self.rank == self.num_stages - 1

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        num_micro_batches: int = 4,
        log_interval: int = 10,
    ):
        """
        Train using pipeline parallelism.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        num_epochs : int
            Number of training epochs
        num_micro_batches : int
            Number of micro-batches per batch
        log_interval : int
            Logging frequency
        """
        logger.info(
            f"[Stage {self.rank}] Starting pipeline training for {num_epochs} epochs"
        )

        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(
                train_loader, epoch, num_micro_batches, log_interval
            )

            if self.is_first_stage:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        num_micro_batches: int,
        log_interval: int,
    ) -> float:
        """
        Train for one epoch using pipeline parallelism.

        This implements the GPipe scheduling algorithm with micro-batches.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        epoch : int
            Current epoch
        num_micro_batches : int
            Number of micro-batches per batch
        log_interval : int
            Logging frequency

        Returns:
        --------
        float : Average loss for epoch
        """
        self.stage.module.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Split batch into micro-batches
            micro_batches = self._split_into_micro_batches(
                data, target, num_micro_batches
            )

            batch_loss = 0.0

            # Process micro-batches in pipeline
            for micro_batch_id, (micro_data, micro_target) in enumerate(micro_batches):
                # Forward pass for this micro-batch
                if self.is_first_stage:
                    # First stage: process input data
                    output = self._forward_micro_batch(micro_data, micro_batch_id)
                else:
                    # Middle/last stages: receive from previous stage
                    input_tensor = self._receive_from_previous_stage()
                    output = self._forward_micro_batch(input_tensor, micro_batch_id)

                # Backward pass
                if self.is_last_stage:
                    # Last stage: compute loss and start backward
                    loss = self.loss_fn(output, micro_target.to(self.stage.device))
                    batch_loss += loss.item()
                    self._backward_micro_batch(loss, micro_batch_id)
                else:
                    # Middle stages: receive gradient from next stage
                    grad_output = self._receive_from_next_stage()
                    self._backward_micro_batch(grad_output, micro_batch_id)

            # Update weights after processing all micro-batches
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Clear stored activations/gradients
            self.micro_batch_activations.clear()
            self.micro_batch_gradients.clear()

            total_loss += batch_loss / num_micro_batches
            num_batches += 1

            # Logging
            if batch_idx % log_interval == 0 and self.is_first_stage:
                logger.info(
                    f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {total_loss / num_batches:.4f}"
                )

        return total_loss / num_batches

    def _split_into_micro_batches(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        num_micro_batches: int,
    ) -> List[tuple]:
        """
        Split batch into micro-batches.

        Parameters:
        -----------
        data : torch.Tensor
            Input data
        target : torch.Tensor
            Target labels
        num_micro_batches : int
            Number of micro-batches

        Returns:
        --------
        List[tuple] : List of (micro_data, micro_target) tuples
        """
        batch_size = data.size(0)
        micro_batch_size = batch_size // num_micro_batches

        micro_batches = []
        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            micro_batches.append((data[start_idx:end_idx], target[start_idx:end_idx]))

        return micro_batches

    def _forward_micro_batch(
        self,
        input_tensor: torch.Tensor,
        micro_batch_id: int,
    ) -> torch.Tensor:
        """
        Forward pass for one micro-batch.

        Parameters:
        -----------
        input_tensor : torch.Tensor
            Input to this stage
        micro_batch_id : int
            Micro-batch identifier

        Returns:
        --------
        torch.Tensor : Output from this stage
        """
        input_tensor = input_tensor.to(self.stage.device)
        output = self.stage.forward(input_tensor)

        # Store activation for backward pass
        output.requires_grad = True
        self.micro_batch_activations.append(output)

        # Send to next stage if not last
        if not self.is_last_stage:
            self._send_to_next_stage(output)

        return output

    def _backward_micro_batch(
        self,
        grad_output: Union[torch.Tensor, None],
        micro_batch_id: int,
    ):
        """
        Backward pass for one micro-batch.

        Parameters:
        -----------
        grad_output : torch.Tensor or None
            Gradient from next stage (or loss for last stage)
        micro_batch_id : int
            Micro-batch identifier
        """
        # Get stored activation
        activation = self.micro_batch_activations.popleft()

        # Compute gradients
        if isinstance(grad_output, torch.Tensor) and grad_output.requires_grad:
            activation.backward(grad_output)
        else:
            # Last stage: grad_output is the loss
            grad_output.backward()

        # Send gradient to previous stage if not first
        if not self.is_first_stage and activation.grad is not None:
            self._send_to_previous_stage(activation.grad)

    def _send_to_next_stage(self, tensor: torch.Tensor):
        """Send tensor to next pipeline stage."""
        if self.is_last_stage:
            return

        # Send to rank + 1
        dist.send(tensor.cpu(), dst=self.rank + 1)

    def _send_to_previous_stage(self, tensor: torch.Tensor):
        """Send gradient to previous pipeline stage."""
        if self.is_first_stage:
            return

        # Send to rank - 1
        dist.send(tensor.cpu(), dst=self.rank - 1)

    def _receive_from_previous_stage(self) -> torch.Tensor:
        """Receive tensor from previous pipeline stage."""
        if self.is_first_stage:
            raise RuntimeError("First stage cannot receive from previous")

        # Receive from rank - 1
        tensor = torch.empty(1)  # Placeholder
        dist.recv(tensor, src=self.rank - 1)
        return tensor

    def _receive_from_next_stage(self) -> torch.Tensor:
        """Receive gradient from next pipeline stage."""
        if self.is_last_stage:
            raise RuntimeError("Last stage cannot receive from next")

        # Receive from rank + 1
        tensor = torch.empty(1)  # Placeholder
        dist.recv(tensor, src=self.rank + 1)
        return tensor


def calculate_pipeline_efficiency(
    num_stages: int,
    num_micro_batches: int,
) -> float:
    """
    Calculate pipeline efficiency (fraction of non-bubble time).

    Efficiency = (Total compute time) / (Total time including bubbles)

    Parameters:
    -----------
    num_stages : int
        Number of pipeline stages
    num_micro_batches : int
        Number of micro-batches

    Returns:
    --------
    float : Efficiency percentage (0-100)
    """
    # In GPipe scheduling:
    # - Bubble time = (num_stages - 1) * time_per_micro_batch
    # - Total time = num_micro_batches * time_per_micro_batch
    #
    # Efficiency = 1 - (bubble_time / total_time)

    if num_micro_batches < num_stages:
        logger.warning(
            f"num_micro_batches ({num_micro_batches}) < num_stages ({num_stages}). "
            "This will have very low efficiency!"
        )

    bubble_time = num_stages - 1
    total_time = num_micro_batches + num_stages - 1

    efficiency = (total_time - bubble_time) / total_time * 100
    return efficiency
