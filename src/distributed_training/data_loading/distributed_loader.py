"""
Efficient Data Loading for Distributed Training

Key optimizations:
- Distributed sampling (each GPU gets different data)
- Prefetching to GPU (overlap data transfer with compute)
- Pinned memory (faster CPU→GPU transfer)
- Multiple workers (parallel data loading)

Author: Your Name
Date: 2025-11
"""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from typing import Optional, Iterator
import threading
import logging

logger = logging.getLogger(__name__)


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create an optimized distributed data loader.

    This automatically sets up DistributedSampler and configures
    optimal settings for distributed training.

    Parameters:
    -----------
    dataset : Dataset
        PyTorch dataset
    batch_size : int
        Batch size per GPU
    num_workers : int
        Number of data loading workers
    pin_memory : bool
        Use pinned memory for faster transfer
    drop_last : bool
        Drop incomplete batches
    shuffle : bool
        Shuffle data
    prefetch_factor : int
        Number of batches to prefetch per worker

    Returns:
    --------
    DataLoader : Configured data loader

    Example:
    --------
    >>> from torchvision import datasets, transforms
    >>> dataset = datasets.CIFAR10(root='./data', train=True,
    ...                            transform=transforms.ToTensor())
    >>> loader = create_distributed_dataloader(
    ...     dataset, batch_size=32, num_workers=4
    ... )
    """
    # Get distributed info
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        logger.warning("Distributed not initialized - using standard DataLoader")

    # Create data loader with optimizations
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,  # Faster CPU→GPU transfer
        drop_last=drop_last,
        shuffle=shuffle,
        prefetch_factor=prefetch_factor,  # Prefetch batches
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
    )

    logger.info(
        f"Created distributed DataLoader: "
        f"batch_size={batch_size}, num_workers={num_workers}, "
        f"pin_memory={pin_memory}"
    )

    return loader


class PrefetchDataLoader:
    """
    Data loader with GPU prefetching.

    This loader prefetches the next batch to GPU while the model
    is processing the current batch, hiding data transfer latency.

    Benefits:
    ---------
    - Overlap data transfer with computation
    - ~10-20% speedup for I/O bound training
    - Minimal memory overhead (1-2 batches)

    Example:
    --------
    >>> base_loader = DataLoader(dataset, batch_size=32)
    >>> prefetch_loader = PrefetchDataLoader(base_loader, device='cuda:0')
    >>> for data, target in prefetch_loader:
    ...     output = model(data)  # Data already on GPU!
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        prefetch_batches: int = 1,
    ):
        """
        Initialize prefetch data loader.

        Parameters:
        -----------
        loader : DataLoader
            Base PyTorch DataLoader
        device : torch.device
            Target GPU device
        prefetch_batches : int
            Number of batches to prefetch (usually 1-2)
        """
        self.loader = loader
        self.device = device
        self.prefetch_batches = prefetch_batches
        self.stream = torch.cuda.Stream()

        logger.info(f"PrefetchDataLoader initialized with device={device}")

    def __iter__(self) -> Iterator:
        """Iterate with prefetching."""
        loader_iter = iter(self.loader)

        # Prefetch first batch
        try:
            next_data, next_target = next(loader_iter)
            next_data = next_data.to(self.device, non_blocking=True)
            next_target = next_target.to(self.device, non_blocking=True)
        except StopIteration:
            return

        # Iterate through batches
        for data, target in loader_iter:
            # Wait for current batch to finish transferring
            torch.cuda.current_stream().wait_stream(self.stream)

            # Yield current batch
            current_data, current_target = next_data, next_target

            # Prefetch next batch in parallel
            with torch.cuda.stream(self.stream):
                next_data = data.to(self.device, non_blocking=True)
                next_target = target.to(self.device, non_blocking=True)

            yield current_data, current_target

        # Yield last batch
        yield next_data, next_target

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)


class BackgroundDataLoader:
    """
    Data loader that loads data in background thread.

    Uses threading to load next batch while model processes current batch.

    Note: For most cases, use PrefetchDataLoader instead. This is useful
    when you need custom background loading logic.
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        queue_size: int = 2,
    ):
        """
        Initialize background data loader.

        Parameters:
        -----------
        loader : DataLoader
            Base PyTorch DataLoader
        device : torch.device
            Target device
        queue_size : int
            Size of prefetch queue
        """
        self.loader = loader
        self.device = device
        self.queue_size = queue_size

    def __iter__(self) -> Iterator:
        """Iterate with background loading."""
        from queue import Queue

        queue = Queue(maxsize=self.queue_size)
        stop_event = threading.Event()

        def producer():
            """Load data in background."""
            for data, target in self.loader:
                if stop_event.is_set():
                    break
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                queue.put((data, target))
            queue.put(None)  # Signal end

        # Start background thread
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                batch = queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            thread.join(timeout=5)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)


def benchmark_dataloader(loader: DataLoader, num_iterations: int = 100) -> dict:
    """
    Benchmark data loader throughput.

    Useful for identifying data loading bottlenecks.

    Parameters:
    -----------
    loader : DataLoader
        Data loader to benchmark
    num_iterations : int
        Number of iterations to run

    Returns:
    --------
    dict : Benchmark statistics

    Example:
    --------
    >>> loader = create_distributed_dataloader(dataset, batch_size=32)
    >>> stats = benchmark_dataloader(loader, num_iterations=100)
    >>> print(f"Throughput: {stats['batches_per_sec']:.2f} batches/s")
    """
    import time

    start_time = time.time()
    total_samples = 0

    for i, (data, target) in enumerate(loader):
        if i >= num_iterations:
            break
        total_samples += data.size(0)

    elapsed_time = time.time() - start_time

    stats = {
        "total_samples": total_samples,
        "total_time_sec": elapsed_time,
        "samples_per_sec": total_samples / elapsed_time,
        "batches_per_sec": num_iterations / elapsed_time,
        "time_per_batch_ms": (elapsed_time / num_iterations) * 1000,
    }

    logger.info(f"DataLoader Benchmark: {stats}")
    return stats
