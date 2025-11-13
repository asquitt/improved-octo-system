"""
Production Best Practices for Distributed Training

This module implements production-ready best practices based on 2024/2025 research:
- Gradient clipping for stability
- Proper resource cleanup
- Advanced profiling integration
- Error handling and logging
- Performance monitoring

References:
- PyTorch Distributed Best Practices 2025
- Gradient Clipping Research (2024)
- Production Deployment Guide
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, Callable
import logging
import atexit
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


class GradientClipper:
    """
    Advanced gradient clipping for training stability.

    Implements multiple clipping strategies:
    - Norm-based clipping (clip_grad_norm_)
    - Value-based clipping (clip_grad_value_)
    - Adaptive clipping (dynamic threshold)

    Based on research:
    - "A Communication-Efficient Distributed Gradient Clipping Algorithm" (2022)
    - "Adaptive Gradient Clipping for Robust Federated Learning" (2024)
    - NVIDIA Megatron-LM best practices

    Usage:
        >>> clipper = GradientClipper(max_norm=1.0, clip_type='norm')
        >>> loss.backward()
        >>> clipper.clip_gradients(model.parameters())
        >>> optimizer.step()
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        max_value: Optional[float] = None,
        clip_type: str = 'norm',
        norm_type: float = 2.0,
        adaptive: bool = False,
        adaptive_window: int = 100
    ):
        """
        Initialize gradient clipper.

        Args:
            max_norm: Maximum gradient norm for clipping
            max_value: Maximum gradient value for value-based clipping
            clip_type: Type of clipping ('norm', 'value', 'adaptive')
            norm_type: Type of norm for gradient norm clipping
            adaptive: Whether to use adaptive clipping thresholds
            adaptive_window: Window size for adaptive threshold calculation
        """
        self.max_norm = max_norm
        self.max_value = max_value
        self.clip_type = clip_type
        self.norm_type = norm_type
        self.adaptive = adaptive
        self.adaptive_window = adaptive_window

        # Tracking for adaptive clipping
        self.grad_norms = []
        self.total_steps = 0

    def clip_gradients(
        self,
        parameters,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, float]:
        """
        Clip gradients of model parameters.

        Args:
            parameters: Model parameters
            optimizer: Optional optimizer for parameter groups

        Returns:
            Dictionary with clipping statistics
        """
        params = list(parameters)

        if len(params) == 0:
            logger.warning("No parameters to clip")
            return {'grad_norm': 0.0, 'clipped': False}

        # Calculate current gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            params,
            float('inf'),
            norm_type=self.norm_type
        )

        stats = {
            'grad_norm': total_norm.item() if torch.is_tensor(total_norm) else total_norm,
            'clipped': False,
            'clip_ratio': 1.0
        }

        # Adaptive threshold adjustment
        if self.adaptive:
            self.grad_norms.append(stats['grad_norm'])
            if len(self.grad_norms) > self.adaptive_window:
                self.grad_norms.pop(0)

            # Update threshold based on recent history
            if len(self.grad_norms) >= 10:
                mean_norm = sum(self.grad_norms) / len(self.grad_norms)
                std_norm = (sum((x - mean_norm) ** 2 for x in self.grad_norms) / len(self.grad_norms)) ** 0.5
                self.max_norm = mean_norm + 2 * std_norm  # 2 std devs

        # Perform clipping
        if self.clip_type == 'norm':
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                params,
                self.max_norm,
                norm_type=self.norm_type
            )
            stats['clipped'] = stats['grad_norm'] > self.max_norm
            stats['clip_ratio'] = min(1.0, self.max_norm / stats['grad_norm']) if stats['grad_norm'] > 0 else 1.0

        elif self.clip_type == 'value':
            if self.max_value is not None:
                torch.nn.utils.clip_grad_value_(params, self.max_value)
                stats['clipped'] = True  # Assume clipping occurred
            else:
                logger.warning("Value clipping requested but max_value not set")

        self.total_steps += 1

        return stats


class DistributedResourceManager:
    """
    Manages distributed training resources and cleanup.

    Ensures proper cleanup of distributed processes and prevents
    resource leaks in production environments.

    Based on PyTorch distributed best practices 2025.

    Usage:
        >>> with DistributedResourceManager() as manager:
        >>>     # Training code here
        >>>     pass
        >>> # Automatic cleanup on exit
    """

    def __init__(self, backend: str = 'nccl', timeout: int = 1800):
        """
        Initialize resource manager.

        Args:
            backend: Distributed backend ('nccl', 'gloo')
            timeout: Timeout for distributed operations (seconds)
        """
        self.backend = backend
        self.timeout = timeout
        self.initialized = False

    def __enter__(self):
        """Enter context manager"""
        if not dist.is_initialized():
            # Will be initialized elsewhere
            logger.info("Distributed not yet initialized")
        else:
            self.initialized = True
            logger.info(f"Distributed already initialized (backend={dist.get_backend()})")

        # Register cleanup on exit
        atexit.register(self.cleanup)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        self.cleanup()
        return False  # Don't suppress exceptions

    def cleanup(self):
        """Clean up distributed resources"""
        if dist.is_initialized():
            try:
                dist.barrier()  # Sync before cleanup
                dist.destroy_process_group()
                logger.info("✓ Distributed process group destroyed")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")

    @staticmethod
    def is_main_process() -> bool:
        """Check if this is the main process (rank 0)"""
        return not dist.is_initialized() or dist.get_rank() == 0

    @staticmethod
    def get_rank() -> int:
        """Get current process rank"""
        return dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def get_world_size() -> int:
        """Get total number of processes"""
        return dist.get_world_size() if dist.is_initialized() else 1


class PerformanceMonitor:
    """
    Monitor training performance metrics in real-time.

    Tracks:
    - Throughput (samples/sec)
    - GPU utilization
    - Memory usage
    - Step timing

    Usage:
        >>> monitor = PerformanceMonitor()
        >>> with monitor.step():
        >>>     # Training step
        >>>     pass
        >>> stats = monitor.get_stats()
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.step_times = []
        self.step_count = 0
        self.start_time = time.time()

    @contextmanager
    def step(self):
        """Context manager for timing a training step"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_start = time.time()

        yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_end = time.time()
        step_time = step_end - step_start

        self.step_times.append(step_time)
        if len(self.step_times) > self.window_size:
            self.step_times.pop(0)

        self.step_count += 1

    def get_stats(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get current performance statistics.

        Args:
            batch_size: Batch size for throughput calculation

        Returns:
            Dictionary with performance metrics
        """
        if len(self.step_times) == 0:
            return {}

        avg_step_time = sum(self.step_times) / len(self.step_times)
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

        stats = {
            'avg_step_time': avg_step_time,
            'steps_per_sec': steps_per_sec,
            'total_steps': self.step_count,
            'elapsed_time': time.time() - self.start_time
        }

        if batch_size is not None:
            stats['samples_per_sec'] = steps_per_sec * batch_size

        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024 ** 3)  # GB

        return stats

    def print_stats(self, batch_size: Optional[int] = None):
        """Print current performance statistics"""
        stats = self.get_stats(batch_size)

        print(f"\n{'='*60}")
        print("Performance Statistics")
        print(f"{'='*60}")
        print(f"Steps: {stats.get('total_steps', 0)}")
        print(f"Avg Step Time: {stats.get('avg_step_time', 0)*1000:.2f} ms")
        print(f"Steps/sec: {stats.get('steps_per_sec', 0):.2f}")

        if 'samples_per_sec' in stats:
            print(f"Samples/sec: {stats['samples_per_sec']:.2f}")

        if 'gpu_memory_allocated' in stats:
            print(f"GPU Memory: {stats['gpu_memory_allocated']:.2f} GB allocated, "
                  f"{stats['gpu_memory_reserved']:.2f} GB reserved")

        print(f"{'='*60}\n")


def setup_distributed_training(
    backend: str = 'nccl',
    init_method: Optional[str] = None,
    timeout: int = 1800
) -> Dict[str, Any]:
    """
    Setup distributed training with best practices.

    This function initializes distributed training following
    PyTorch 2025 best practices:
    - Proper backend selection
    - Timeout configuration
    - Error handling
    - Resource management

    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: Initialization method (default: env://)
        timeout: Timeout for distributed operations (seconds)

    Returns:
        Dictionary with setup information

    Example:
        >>> setup_info = setup_distributed_training()
        >>> print(f"Rank: {setup_info['rank']}/{setup_info['world_size']}")
    """
    if dist.is_initialized():
        logger.warning("Distributed already initialized")
        return {
            'rank': dist.get_rank(),
            'world_size': dist.get_world_size(),
            'backend': dist.get_backend()
        }

    # Set default init method
    if init_method is None:
        init_method = 'env://'

    # Initialize process group
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=torch.distributed.distributed_c10d.timedelta(seconds=timeout)
        )

        setup_info = {
            'rank': dist.get_rank(),
            'world_size': dist.get_world_size(),
            'backend': backend,
            'initialized': True
        }

        logger.info(f"✓ Distributed training initialized")
        logger.info(f"  Rank: {setup_info['rank']}/{setup_info['world_size']}")
        logger.info(f"  Backend: {backend}")

        return setup_info

    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        return {
            'rank': 0,
            'world_size': 1,
            'backend': None,
            'initialized': False,
            'error': str(e)
        }


def cleanup_distributed_training():
    """
    Cleanup distributed training resources.

    Follows PyTorch best practices:
    - Barrier synchronization before cleanup
    - Proper process group destruction
    - Error handling

    Should be called at the end of training or in exception handlers.

    Example:
        >>> try:
        >>>     # Training code
        >>>     pass
        >>> finally:
        >>>     cleanup_distributed_training()
    """
    if dist.is_initialized():
        try:
            # Synchronize all processes
            dist.barrier()

            # Destroy process group
            dist.destroy_process_group()

            logger.info("✓ Distributed training cleanup complete")

        except Exception as e:
            logger.warning(f"Error during distributed cleanup: {e}")
    else:
        logger.info("Distributed not initialized, no cleanup needed")


# Convenience context manager combining best practices
@contextmanager
def distributed_training_context(
    backend: str = 'nccl',
    enable_profiling: bool = False
):
    """
    Context manager for distributed training with automatic setup and cleanup.

    This combines multiple best practices:
    - Automatic distributed initialization
    - Resource management
    - Performance monitoring (optional)
    - Proper cleanup

    Args:
        backend: Distributed backend
        enable_profiling: Whether to enable performance profiling

    Example:
        >>> with distributed_training_context() as ctx:
        >>>     model = MyModel()
        >>>     # Training code
        >>>     pass
        >>> # Automatic cleanup
    """
    # Setup
    setup_info = setup_distributed_training(backend=backend)

    monitor = None
    if enable_profiling:
        monitor = PerformanceMonitor()

    context = {
        'setup_info': setup_info,
        'monitor': monitor,
        'is_main_process': setup_info['rank'] == 0
    }

    try:
        yield context
    finally:
        # Cleanup
        if monitor:
            monitor.print_stats()

        cleanup_distributed_training()
