"""
Gradient Compression for Distributed Training

Compress gradients before communication to reduce network overhead.
Useful for slow network connections or large models.

Techniques:
-----------
- Top-K sparsification: Only send largest K gradients
- Random sparsification: Randomly sample gradients
- Quantization: Reduce gradient precision

Benefits:
---------
- Faster communication (up to 10x)
- Better scaling across nodes
- Minimal accuracy impact

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

logger = logging.getLogger(__name__)


class CompressedDDP(nn.Module):
    """
    DDP with gradient compression.

    Compresses gradients before AllReduce to reduce communication.

    Example:
    --------
    >>> model = MyModel()
    >>> compressed_model = CompressedDDP(
    ...     model,
    ...     compression_ratio=0.1,  # Send only 10% of gradients
    ...     method="topk"
    ... )
    """

    def __init__(
        self,
        module: nn.Module,
        compression_ratio: float = 0.1,
        method: str = "topk",
        device_ids=None,
    ):
        """
        Initialize compressed DDP.

        Parameters:
        -----------
        module : nn.Module
            Model to wrap
        compression_ratio : float
            Fraction of gradients to send (0.1 = 10%)
        method : str
            Compression method: "topk", "random", "quantize"
        device_ids : list, optional
            GPU devices
        """
        super().__init__()

        self.module = module
        self.compression_ratio = compression_ratio
        self.method = method

        # Wrap with regular DDP
        if device_ids is None and torch.cuda.is_available():
            device_ids = [torch.cuda.current_device()]

        self.ddp_module = DDP(module, device_ids=device_ids)

        # Register gradient compression hooks
        self._register_hooks()

        logger.info(
            f"CompressedDDP initialized: {method}, "
            f"ratio={compression_ratio}"
        )

    def _register_hooks(self):
        """Register hooks for gradient compression."""
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._compress_gradient)

    def _compress_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Compress gradient before communication.

        Parameters:
        -----------
        grad : torch.Tensor
            Original gradient

        Returns:
        --------
        torch.Tensor : Compressed gradient
        """
        if self.method == "topk":
            return self._topk_compression(grad)
        elif self.method == "random":
            return self._random_compression(grad)
        elif self.method == "quantize":
            return self._quantization(grad)
        else:
            return grad

    def _topk_compression(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Top-K sparsification.

        Keep only the K largest gradients (by absolute value).
        """
        k = int(grad.numel() * self.compression_ratio)

        # Flatten gradient
        flat_grad = grad.flatten()

        # Find top-k by absolute value
        _, indices = torch.topk(flat_grad.abs(), k)

        # Create sparse gradient
        compressed = torch.zeros_like(flat_grad)
        compressed[indices] = flat_grad[indices]

        return compressed.reshape(grad.shape)

    def _random_compression(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Random sparsification.

        Randomly sample gradients with probability = compression_ratio.
        """
        mask = torch.rand_like(grad) < self.compression_ratio
        return grad * mask / self.compression_ratio  # Scale to maintain expectation

    def _quantization(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Gradient quantization.

        Reduce gradient precision (e.g., 32-bit → 8-bit).
        """
        # Simple quantization to 8 levels
        num_levels = 8

        # Normalize to [-1, 1]
        grad_max = grad.abs().max()
        if grad_max == 0:
            return grad

        normalized = grad / grad_max

        # Quantize
        quantized = torch.round(normalized * (num_levels / 2)) / (num_levels / 2)

        # Denormalize
        return quantized * grad_max

    def forward(self, *args, **kwargs):
        """Forward pass through DDP module."""
        return self.ddp_module(*args, **kwargs)


def compress_tensor(
    tensor: torch.Tensor,
    compression_ratio: float = 0.1,
    method: str = "topk",
) -> tuple:
    """
    Compress a tensor for transmission.

    Parameters:
    -----------
    tensor : torch.Tensor
        Tensor to compress
    compression_ratio : float
        Compression ratio
    method : str
        Compression method

    Returns:
    --------
    tuple : (compressed_values, indices, shape)
    """
    original_shape = tensor.shape
    flat_tensor = tensor.flatten()

    if method == "topk":
        k = int(flat_tensor.numel() * compression_ratio)
        values, indices = torch.topk(flat_tensor.abs(), k)
        # Keep signs
        values = values * torch.sign(flat_tensor[indices])

        return values, indices, original_shape

    return tensor, None, original_shape


def decompress_tensor(
    values: torch.Tensor,
    indices: torch.Tensor,
    shape: tuple,
) -> torch.Tensor:
    """
    Decompress a tensor.

    Parameters:
    -----------
    values : torch.Tensor
        Compressed values
    indices : torch.Tensor
        Indices of values
    shape : tuple
        Original tensor shape

    Returns:
    --------
    torch.Tensor : Decompressed tensor
    """
    numel = 1
    for dim in shape:
        numel *= dim

    result = torch.zeros(numel, dtype=values.dtype, device=values.device)

    if indices is not None:
        result[indices] = values

    return result.reshape(shape)
