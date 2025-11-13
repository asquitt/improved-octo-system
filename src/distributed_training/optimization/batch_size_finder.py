"""
Automatic Batch Size Finder

Automatically find the largest batch size that fits in GPU memory.
Uses binary search to efficiently find optimal batch size.

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


def find_optimal_batch_size(
    model: nn.Module,
    input_shape: tuple,
    max_batch_size: int = 1024,
    min_batch_size: int = 1,
    num_iterations: int = 5,
    device: str = "cuda",
) -> int:
    """
    Find optimal batch size using binary search.

    Parameters:
    -----------
    model : nn.Module
        Model to test
    input_shape : tuple
        Input tensor shape (without batch dimension)
    max_batch_size : int
        Maximum batch size to try
    min_batch_size : int
        Minimum batch size
    num_iterations : int
        Number of forward/backward passes to test
    device : str
        Device to use

    Returns:
    --------
    int : Optimal batch size

    Example:
    --------
    >>> model = MyModel()
    >>> optimal_bs = find_optimal_batch_size(
    ...     model,
    ...     input_shape=(3, 224, 224)
    ... )
    >>> print(f"Use batch size: {optimal_bs}")
    """
    logger.info(f"Finding optimal batch size in range [{min_batch_size}, {max_batch_size}]")

    model = model.to(device)
    model.train()

    def can_fit_batch_size(batch_size: int) -> bool:
        """Test if batch size fits in memory."""
        try:
            # Clear cache
            torch.cuda.empty_cache()

            # Create dummy data
            dummy_input = torch.randn(batch_size, *input_shape).to(device)
            dummy_target = torch.randint(0, 10, (batch_size,)).to(device)

            # Test forward and backward
            for _ in range(num_iterations):
                output = model(dummy_input)
                loss = nn.CrossEntropyLoss()(output, dummy_target)
                loss.backward()

            # If we got here, it fits
            del dummy_input, dummy_target, output, loss
            torch.cuda.empty_cache()
            return True

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return False
            raise

    # Binary search
    left, right = min_batch_size, max_batch_size
    optimal = min_batch_size

    while left <= right:
        mid = (left + right) // 2

        logger.info(f"Testing batch size: {mid}")

        if can_fit_batch_size(mid):
            optimal = mid
            left = mid + 1
            logger.info(f"✓ Batch size {mid} fits")
        else:
            right = mid - 1
            logger.info(f"✗ Batch size {mid} OOM")

    logger.info(f"Optimal batch size: {optimal}")

    # Add safety margin (use 80% of maximum)
    safe_batch_size = int(optimal * 0.8)
    logger.info(f"Recommended batch size (with safety margin): {safe_batch_size}")

    return safe_batch_size
