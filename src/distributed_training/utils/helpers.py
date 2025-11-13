"""
Utility functions for distributed training.

Common helpers for setup, debugging, and convenience.

Author: Your Name
Date: 2025-11
"""

import torch
import torch.distributed as dist
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    - CuDNN (deterministic mode)

    Parameters:
    -----------
    seed : int
        Random seed

    Example:
    --------
    >>> set_seed(42)  # Reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")


def get_rank() -> int:
    """
    Get current process rank.

    Returns:
    --------
    int : Process rank (0 if not distributed)
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get total number of processes.

    Returns:
    --------
    int : World size (1 if not distributed)
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """
    Check if current process is main (rank 0).

    Returns:
    --------
    bool : True if main process
    """
    return get_rank() == 0


def print_rank_0(message: str):
    """Print message only on rank 0."""
    if is_main_process():
        print(message)


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
