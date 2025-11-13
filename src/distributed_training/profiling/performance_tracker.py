"""
Performance Profiling for Distributed Training

Track GPU utilization, throughput, memory usage, and bottlenecks.

Author: Your Name
Date: 2025-11
"""

import torch
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Track training performance metrics.

    Monitors:
    - GPU utilization and memory
    - Training throughput (samples/sec)
    - Time breakdown (data loading vs compute)
    - Communication overhead

    Example:
    --------
    >>> tracker = PerformanceTracker()
    >>> with tracker.measure("forward"):
    ...     output = model(data)
    >>> stats = tracker.get_stats()
    >>> print(f"Forward pass: {stats['forward']['avg_ms']:.2f}ms")
    """

    def __init__(self):
        """Initialize performance tracker."""
        self.timers = defaultdict(list)
        self.current_timers = {}
        self.sample_counts = []
        self.start_time = time.time()

        logger.info("PerformanceTracker initialized")

    def start_timer(self, name: str):
        """Start a named timer."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.current_timers[name] = time.time()

    def stop_timer(self, name: str):
        """Stop a named timer and record duration."""
        if name not in self.current_timers:
            logger.warning(f"Timer '{name}' was not started")
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration = time.time() - self.current_timers[name]
        self.timers[name].append(duration)
        del self.current_timers[name]

    def measure(self, name: str):
        """
        Context manager for measuring execution time.

        Example:
        --------
        >>> with tracker.measure("data_loading"):
        ...     data, target = next(dataloader)
        """
        class TimerContext:
            def __init__(ctx_self, tracker, name):
                ctx_self.tracker = tracker
                ctx_self.name = name

            def __enter__(ctx_self):
                ctx_self.tracker.start_timer(ctx_self.name)
                return ctx_self

            def __exit__(ctx_self, *args):
                ctx_self.tracker.stop_timer(ctx_self.name)

        return TimerContext(self, name)

    def record_samples(self, num_samples: int):
        """Record number of samples processed."""
        self.sample_counts.append(num_samples)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics.

        Returns:
        --------
        dict : Statistics for each timer
        """
        stats = {}

        for name, times in self.timers.items():
            if not times:
                continue

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)

            stats[name] = {
                "avg_ms": avg_time * 1000,
                "min_ms": min_time * 1000,
                "max_ms": max_time * 1000,
                "total_sec": total_time,
                "count": len(times),
            }

        # Calculate throughput
        if self.sample_counts:
            total_samples = sum(self.sample_counts)
            elapsed = time.time() - self.start_time
            stats["throughput"] = {
                "samples_per_sec": total_samples / elapsed if elapsed > 0 else 0,
                "total_samples": total_samples,
                "total_time_sec": elapsed,
            }

        return stats

    def get_gpu_stats(self) -> Dict[str, float]:
        """
        Get current GPU statistics.

        Returns:
        --------
        dict : GPU memory and utilization stats
        """
        if not torch.cuda.is_available():
            return {}

        stats = {}
        device = torch.cuda.current_device()

        # Memory stats
        stats["memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
        stats["memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
        stats["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9

        # Try to get utilization (requires nvidia-smi)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats["gpu_utilization_percent"] = util.gpu
            stats["memory_utilization_percent"] = util.memory
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return stats

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("Performance Statistics")
        print("=" * 60)

        for name, metrics in stats.items():
            if name == "throughput":
                print(f"\n{name.upper()}:")
                print(f"  Samples/sec: {metrics['samples_per_sec']:.2f}")
                print(f"  Total samples: {metrics['total_samples']}")
                print(f"  Total time: {metrics['total_time_sec']:.2f}s")
            else:
                print(f"\n{name}:")
                print(f"  Average: {metrics['avg_ms']:.2f}ms")
                print(f"  Min: {metrics['min_ms']:.2f}ms")
                print(f"  Max: {metrics['max_ms']:.2f}ms")
                print(f"  Count: {metrics['count']}")

        # GPU stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print(f"\nGPU STATS:")
            for key, value in gpu_stats.items():
                print(f"  {key}: {value:.2f}")

        print("=" * 60 + "\n")

    def save_stats(self, filepath: str):
        """Save statistics to JSON file."""
        stats = {
            "timers": self.get_stats(),
            "gpu": self.get_gpu_stats(),
        }

        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved performance stats to {filepath}")

    def reset(self):
        """Reset all statistics."""
        self.timers.clear()
        self.current_timers.clear()
        self.sample_counts.clear()
        self.start_time = time.time()


def profile_model(
    model: torch.nn.Module,
    input_shape: tuple,
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Profile model to get FLOPs, parameters, and memory usage.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to profile
    input_shape : tuple
        Input tensor shape (without batch dimension)
    device : str
        Device to run on

    Returns:
    --------
    dict : Profiling results

    Example:
    --------
    >>> model = MyModel()
    >>> stats = profile_model(model, input_shape=(3, 224, 224))
    >>> print(f"Parameters: {stats['total_params']:,}")
    """
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Memory usage
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # Forward pass timing
    dummy_input = torch.randn(1, *input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    avg_time_ms = (elapsed / 100) * 1000

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_memory_mb": param_memory / (1024 ** 2),
        "inference_time_ms": avg_time_ms,
        "throughput_samples_per_sec": 1000 / avg_time_ms,
    }

    # Try to get FLOPs with thop if available
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
        stats["flops"] = flops
        stats["gflops"] = flops / 1e9
    except ImportError:
        pass

    return stats
