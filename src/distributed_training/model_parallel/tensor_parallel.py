"""
Tensor Parallelism Implementation

This module implements tensor parallelism (also called model parallelism),
where a single model is split across multiple GPUs.

Key Concepts:
-------------
1. **Tensor Parallelism**: Split weight tensors across GPUs
2. **Column Parallel**: Split output dimension (columns) of weight matrix
3. **Row Parallel**: Split input dimension (rows) of weight matrix
4. **Communication**: Use AllReduce/AllGather for tensor synchronization

Why Use Model Parallelism?
---------------------------
- Model is too large for single GPU memory
- Want to train models with billions of parameters
- Can be combined with data parallelism for hybrid approach

Example:
--------
For a linear layer: Y = XW + b
- Column Parallel: Split W vertically → [W1 | W2] on different GPUs
- Row Parallel: Split W horizontally → [W1] [W2] on different GPUs

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    Splits the weight matrix along the output dimension (columns).

    Example:
    --------
    Input: X [batch, in_features]
    Weight: W [in_features, out_features]

    Column Parallel splits W vertically:
    GPU 0: W1 [in_features, out_features/2]
    GPU 1: W2 [in_features, out_features/2]

    Each GPU computes:
    GPU 0: Y1 = X @ W1
    GPU 1: Y2 = X @ W2

    Output is concatenated: Y = [Y1 | Y2]

    Benefits:
    ---------
    - Reduces memory per GPU (weight matrix is split)
    - No communication needed during forward pass
    - Output needs AllGather if feeding to non-parallel layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
    ):
        """
        Initialize column parallel linear layer.

        Parameters:
        -----------
        in_features : int
            Size of input features
        out_features : int
            Size of output features (will be split across GPUs)
        bias : bool
            Whether to use bias term
        gather_output : bool
            Whether to gather output across GPUs (for feeding to non-parallel layer)
        """
        super().__init__()

        # Get distributed training info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.gather_output = gather_output

        # Calculate this GPU's output features
        assert out_features % self.world_size == 0, (
            f"out_features ({out_features}) must be divisible by "
            f"world_size ({self.world_size})"
        )
        self.out_features_per_partition = out_features // self.world_size

        # Create weight matrix (only this GPU's partition)
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

        # Create bias (only this GPU's partition)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        self._initialize_weights()

        logger.debug(
            f"[Rank {self.rank}] ColumnParallelLinear: "
            f"in={in_features}, out={self.out_features_per_partition}"
        )

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with column parallelism.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch, in_features]

        Returns:
        --------
        torch.Tensor
            Output tensor [batch, out_features_per_partition]
            Or [batch, out_features] if gather_output=True
        """
        # Linear transformation: Y = XW^T + b
        output = torch.matmul(x, self.weight.t())

        if self.bias is not None:
            output = output + self.bias

        # Gather outputs from all GPUs if requested
        if self.gather_output:
            output = self._gather_from_parallel_region(output)

        return output

    def _gather_from_parallel_region(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all GPUs and concatenate.

        Uses AllGather collective operation.
        """
        if self.world_size == 1:
            return tensor

        # Allocate space for all partitions
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]

        # Gather from all GPUs
        dist.all_gather(tensor_list, tensor)

        # Concatenate along last dimension
        output = torch.cat(tensor_list, dim=-1)
        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    Splits the weight matrix along the input dimension (rows).

    Example:
    --------
    Input: X [batch, in_features]
    Weight: W [in_features, out_features]

    Row Parallel splits W horizontally:
    GPU 0: W1 [in_features/2, out_features]
    GPU 1: W2 [in_features/2, out_features]

    Each GPU computes partial output:
    GPU 0: Y1 = X1 @ W1
    GPU 1: Y2 = X2 @ W2

    Final output: Y = Y1 + Y2 (via AllReduce)

    Benefits:
    ---------
    - Reduces memory per GPU (weight matrix is split)
    - Complements column parallel layer
    - Requires AllReduce for final output
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
    ):
        """
        Initialize row parallel linear layer.

        Parameters:
        -----------
        in_features : int
            Size of input features (will be split across GPUs)
        out_features : int
            Size of output features
        bias : bool
            Whether to use bias term
        input_is_parallel : bool
            Whether input is already partitioned across GPUs
        """
        super().__init__()

        # Get distributed training info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.input_is_parallel = input_is_parallel

        # Calculate this GPU's input features
        assert in_features % self.world_size == 0, (
            f"in_features ({in_features}) must be divisible by "
            f"world_size ({self.world_size})"
        )
        self.in_features_per_partition = in_features // self.world_size

        # Create weight matrix (only this GPU's partition)
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        # Bias only on first GPU to avoid duplication
        if bias and self.rank == 0:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        self._initialize_weights()

        logger.debug(
            f"[Rank {self.rank}] RowParallelLinear: "
            f"in={self.in_features_per_partition}, out={out_features}"
        )

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with row parallelism.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch, in_features] or [batch, in_features_per_partition]

        Returns:
        --------
        torch.Tensor
            Output tensor [batch, out_features]
        """
        # Split input if not already parallel
        if not self.input_is_parallel:
            x = self._scatter_to_parallel_region(x)

        # Linear transformation: Y = XW^T
        output = torch.matmul(x, self.weight.t())

        # AllReduce to sum partial results from all GPUs
        output = self._reduce_from_parallel_region(output)

        # Add bias (only on rank 0 to avoid duplication)
        if self.bias is not None:
            output = output + self.bias

        return output

    def _scatter_to_parallel_region(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Scatter tensor to parallel region (split across GPUs).
        """
        if self.world_size == 1:
            return tensor

        # Split tensor along last dimension
        tensor_list = torch.split(tensor, self.in_features_per_partition, dim=-1)

        # Get this GPU's partition
        output = tensor_list[self.rank].contiguous()
        return output

    def _reduce_from_parallel_region(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reduce tensor from parallel region using AllReduce.

        Sums partial results from all GPUs.
        """
        if self.world_size == 1:
            return tensor

        # AllReduce with SUM operation
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor


class TensorParallelModel(nn.Module):
    """
    Example model with tensor parallelism.

    This demonstrates how to build a model with column and row parallel layers.

    Architecture:
    -------------
    Input → Column Parallel → Activation → Row Parallel → Output

    The column and row parallel layers work together:
    - Column parallel splits output dimension
    - Row parallel expects split input
    - No communication between them!
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
    ):
        """
        Initialize tensor parallel model.

        Parameters:
        -----------
        input_size : int
            Input feature dimension
        hidden_size : int
            Hidden layer dimension (will be split across GPUs)
        output_size : int
            Output dimension
        num_layers : int
            Number of hidden layers
        """
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # First layer: column parallel
        self.layers.append(
            ColumnParallelLinear(
                in_features=input_size,
                out_features=hidden_size,
                bias=True,
                gather_output=False,  # Output stays split
            )
        )

        # Hidden layers: alternating column and row parallel
        for i in range(num_layers - 1):
            # Row parallel (takes split input, produces full output)
            self.layers.append(
                RowParallelLinear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    bias=True,
                    input_is_parallel=True,  # Input is already split
                )
            )

            # Column parallel (takes full input, produces split output)
            self.layers.append(
                ColumnParallelLinear(
                    in_features=hidden_size,
                    out_features=hidden_size,
                    bias=True,
                    gather_output=False,  # Keep split for next layer
                )
            )

        # Final layer: row parallel (gather output)
        self.layers.append(
            RowParallelLinear(
                in_features=hidden_size,
                out_features=output_size,
                bias=True,
                input_is_parallel=True,
            )
        )

        self.activation = nn.ReLU()

        logger.info(
            f"TensorParallelModel initialized with {len(self.layers)} layers"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through tensor parallel model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor [batch, input_size]

        Returns:
        --------
        torch.Tensor
            Output tensor [batch, output_size]
        """
        for layer in self.layers:
            x = layer(x)
            # Apply activation after each layer except last
            if layer != self.layers[-1]:
                x = self.activation(x)

        return x


def get_model_memory_usage(model: nn.Module) -> dict:
    """
    Calculate memory usage of model parameters.

    Useful for understanding memory savings with model parallelism.

    Parameters:
    -----------
    model : nn.Module
        The model to analyze

    Returns:
    --------
    dict : Memory usage statistics
    """
    param_size = 0
    param_count = 0

    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    buffer_count = 0

    for buffer in model.buffers():
        buffer_count += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size
    size_mb = total_size / (1024 ** 2)
    size_gb = total_size / (1024 ** 3)

    return {
        "param_count": param_count,
        "buffer_count": buffer_count,
        "param_size_mb": param_size / (1024 ** 2),
        "buffer_size_mb": buffer_size / (1024 ** 2),
        "total_size_mb": size_mb,
        "total_size_gb": size_gb,
    }
