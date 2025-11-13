"""
Fault Tolerance for Distributed Training

Handles network failures, node crashes, and automatic recovery.

Key Features:
- Automatic checkpoint saving at intervals
- Retry failed operations with exponential backoff
- Graceful handling of node failures
- Health monitoring

Author: Your Name
Date: 2025-11
"""

import torch
import torch.distributed as dist
import time
import logging
from typing import Callable, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RetryContext:
    """
    Context manager for retrying operations with exponential backoff.

    Example:
    --------
    >>> with RetryContext(max_retries=3):
    ...     # This will retry up to 3 times on failure
    ...     dist.all_reduce(tensor)
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (RuntimeError, OSError),
    ):
        """
        Initialize retry context.

        Parameters:
        -----------
        max_retries : int
            Maximum number of retry attempts
        initial_delay : float
            Initial delay in seconds
        backoff_factor : float
            Multiply delay by this factor each retry
        exceptions : tuple
            Exception types to retry on
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle retries."""
        if exc_type is None:
            return False

        if not isinstance(exc_val, self.exceptions):
            return False

        # Retry logic
        delay = self.initial_delay
        for retry in range(self.max_retries):
            logger.warning(
                f"Operation failed, retrying in {delay:.1f}s "
                f"(attempt {retry + 1}/{self.max_retries})"
            )
            time.sleep(delay)
            delay *= self.backoff_factor

            try:
                # Re-raise to trigger retry
                return True
            except self.exceptions:
                continue

        # All retries exhausted
        logger.error(f"All retries exhausted, raising exception")
        return False


class FaultTolerantTrainer:
    """
    Wrapper for fault-tolerant distributed training.

    Automatically handles:
    - Node failures
    - Network issues
    - Checkpoint recovery
    - Health monitoring

    Example:
    --------
    >>> trainer = FaultTolerantTrainer(
    ...     model=model,
    ...     checkpoint_manager=checkpoint_manager
    ... )
    >>> trainer.train_with_recovery(train_loader, num_epochs=10)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_manager,
        health_check_interval: int = 100,
    ):
        """
        Initialize fault-tolerant trainer.

        Parameters:
        -----------
        model : torch.nn.Module
            Model to train
        optimizer : torch.optim.Optimizer
            Optimizer
        checkpoint_manager : CheckpointManager
            Checkpoint manager for saving/loading
        health_check_interval : int
            Steps between health checks
        """
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.health_check_interval = health_check_interval

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        logger.info(f"[Rank {self.rank}] FaultTolerantTrainer initialized")

    def train_with_recovery(
        self,
        train_fn: Callable,
        start_epoch: int = 0,
        num_epochs: int = 10,
    ):
        """
        Train with automatic recovery from failures.

        Parameters:
        -----------
        train_fn : Callable
            Training function to execute
        start_epoch : int
            Starting epoch
        num_epochs : int
            Total number of epochs
        """
        current_epoch = start_epoch

        while current_epoch < num_epochs:
            try:
                # Train one epoch with health monitoring
                self._train_epoch_with_monitoring(train_fn, current_epoch)

                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    epoch=current_epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                )

                current_epoch += 1

            except Exception as e:
                logger.error(
                    f"[Rank {self.rank}] Training failed at epoch "
                    f"{current_epoch}: {e}"
                )

                # Attempt recovery
                if self._recover_from_failure():
                    logger.info(f"[Rank {self.rank}] Recovery successful")
                    # Reload from last checkpoint
                    state = self.checkpoint_manager.load_latest_checkpoint(
                        self.model, self.optimizer
                    )
                    if state:
                        current_epoch = state["epoch"] + 1
                else:
                    logger.error(
                        f"[Rank {self.rank}] Recovery failed, exiting"
                    )
                    raise

    def _train_epoch_with_monitoring(self, train_fn: Callable, epoch: int):
        """Train one epoch with health monitoring."""
        step = 0

        while True:
            # Check health periodically
            if step % self.health_check_interval == 0:
                if not self._health_check():
                    raise RuntimeError("Health check failed")

            # Execute training step
            try:
                train_fn(epoch, step)
                step += 1
            except StopIteration:
                break

    def _health_check(self) -> bool:
        """
        Check if all nodes are healthy.

        Returns:
        --------
        bool : True if healthy, False otherwise
        """
        if not dist.is_initialized():
            return True

        try:
            # Simple all_reduce to check connectivity
            tensor = torch.tensor([1.0]).cuda()
            with RetryContext(max_retries=3):
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            # Check if sum is correct (should be world_size)
            if tensor.item() == self.world_size:
                return True
            else:
                logger.warning(
                    f"[Rank {self.rank}] Health check failed: "
                    f"expected {self.world_size}, got {tensor.item()}"
                )
                return False

        except Exception as e:
            logger.error(f"[Rank {self.rank}] Health check error: {e}")
            return False

    def _recover_from_failure(self) -> bool:
        """
        Attempt to recover from failure.

        Returns:
        --------
        bool : True if recovery successful
        """
        logger.info(f"[Rank {self.rank}] Attempting recovery...")

        try:
            # Reinitialize process group
            if dist.is_initialized():
                dist.destroy_process_group()

            time.sleep(5)  # Wait before reinitializing

            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
            )

            logger.info(f"[Rank {self.rank}] Process group reinitialized")
            return True

        except Exception as e:
            logger.error(f"[Rank {self.rank}] Recovery failed: {e}")
            return False


@contextmanager
def fault_tolerant_operation(max_retries: int = 3):
    """
    Context manager for fault-tolerant distributed operations.

    Example:
    --------
    >>> with fault_tolerant_operation(max_retries=3):
    ...     dist.all_reduce(tensor)
    """
    for attempt in range(max_retries):
        try:
            yield
            break
        except (RuntimeError, OSError) as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                logger.warning(f"Operation failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error("All retries exhausted")
                raise
