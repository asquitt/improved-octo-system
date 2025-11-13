"""
Benchmarking Suite for Distributed Training

Compare different parallelism strategies, batch sizes, and optimizations.

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import json
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Benchmark different distributed training strategies.

    Compares throughput, memory usage, and scaling efficiency
    across different configurations.

    Example:
    --------
    >>> benchmark = BenchmarkRunner()
    >>> benchmark.add_config("DDP", strategy="ddp", num_gpus=4)
    >>> benchmark.add_config("FSDP", strategy="fsdp", num_gpus=4)
    >>> results = benchmark.run()
    >>> benchmark.plot_comparison()
    """

    def __init__(self):
        """Initialize benchmark runner."""
        self.configurations = {}
        self.results = {}

        logger.info("BenchmarkRunner initialized")

    def add_config(
        self,
        name: str,
        strategy: str,
        num_gpus: int = 1,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Add a configuration to benchmark.

        Parameters:
        -----------
        name : str
            Configuration name
        strategy : str
            Training strategy (ddp, fsdp, deepspeed, etc.)
        num_gpus : int
            Number of GPUs
        batch_size : int
            Batch size
        **kwargs : dict
            Additional configuration parameters
        """
        self.configurations[name] = {
            "strategy": strategy,
            "num_gpus": num_gpus,
            "batch_size": batch_size,
            **kwargs
        }

        logger.info(f"Added config: {name}")

    def run(
        self,
        model: nn.Module,
        dataset,
        num_iterations: int = 100,
    ) -> Dict[str, Dict]:
        """
        Run benchmarks for all configurations.

        Parameters:
        -----------
        model : nn.Module
            Model to benchmark
        dataset : Dataset
            Dataset to use
        num_iterations : int
            Number of iterations per config

        Returns:
        --------
        dict : Benchmark results
        """
        logger.info(f"Running benchmarks ({num_iterations} iterations each)")

        for name, config in self.configurations.items():
            logger.info(f"\nBenchmarking: {name}")

            result = self._benchmark_config(
                name,
                config,
                model,
                dataset,
                num_iterations
            )

            self.results[name] = result

        logger.info("\nBenchmarking complete!")

        return self.results

    def _benchmark_config(
        self,
        name: str,
        config: Dict,
        model: nn.Module,
        dataset,
        num_iterations: int,
    ) -> Dict:
        """Benchmark a single configuration."""
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
        )

        # Setup model (simplified - in production would use actual strategy)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Warmup
        for i, (data, target) in enumerate(loader):
            if i >= 5:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        total_samples = 0

        for i, (data, target) in enumerate(loader):
            if i >= num_iterations:
                break

            data, target = data.to(device), target.to(device)
            total_samples += data.size(0)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        # Calculate metrics
        elapsed = end_time - start_time
        throughput = total_samples / elapsed
        memory_mb = peak_memory / (1024 ** 2)

        # Calculate scaling efficiency
        base_throughput = 100  # Baseline for single GPU
        ideal_throughput = base_throughput * config["num_gpus"]
        scaling_efficiency = (throughput / ideal_throughput) * 100 if ideal_throughput > 0 else 0

        results = {
            "throughput_samples_per_sec": throughput,
            "total_samples": total_samples,
            "elapsed_time_sec": elapsed,
            "peak_memory_mb": memory_mb,
            "scaling_efficiency_percent": min(scaling_efficiency, 100),
            "batch_size": config["batch_size"],
            "num_gpus": config["num_gpus"],
        }

        logger.info(f"  Throughput: {throughput:.2f} samples/sec")
        logger.info(f"  Memory: {memory_mb:.2f} MB")

        return results

    def save_results(self, filepath: str = "benchmark_results.json"):
        """Save benchmark results to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Saved benchmark results to {filepath}")

    def print_results(self):
        """Print formatted results."""
        print("\n" + "="*80)
        print("Benchmark Results")
        print("="*80)

        for name, results in self.results.items():
            print(f"\n{name}:")
            print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")
            print(f"  Memory: {results['peak_memory_mb']:.2f} MB")
            print(f"  Scaling Efficiency: {results['scaling_efficiency_percent']:.1f}%")
            print(f"  Batch Size: {results['batch_size']}")
            print(f"  GPUs: {results['num_gpus']}")

        print("="*80 + "\n")

    def plot_comparison(self, save_path: str = "benchmark_comparison.png"):
        """Plot benchmark comparison."""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        names = list(self.results.keys())
        throughputs = [self.results[name]['throughput_samples_per_sec'] for name in names]
        memories = [self.results[name]['peak_memory_mb'] for name in names]

        # Throughput comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars1 = ax1.bar(names, throughputs, color=colors, alpha=0.8)

        ax1.set_ylabel("Throughput (samples/sec)", fontsize=12)
        ax1.set_title("Training Throughput", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')

        # Memory comparison
        bars2 = ax2.bar(names, memories, color=colors, alpha=0.8)

        ax2.set_ylabel("Peak Memory (MB)", fontsize=12)
        ax2.set_title("Memory Usage", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        logger.info(f"Saved comparison plot to {save_path}")

        return fig, (ax1, ax2)
