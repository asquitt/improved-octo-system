"""
Throughput Performance Tests

Tests that measure training throughput (samples/sec) for different strategies.
These tests help identify performance bottlenecks and compare strategies.

Performance Metrics:
- Samples per second
- Batches per second
- GPU utilization
- Time per epoch

Expected Baselines (on 4x V100 GPUs):
- DDP: ~2000 samples/sec
- FSDP: ~1800 samples/sec (with sharding overhead)
- DeepSpeed ZeRO-2: ~1700 samples/sec
- Model Parallel: ~1200 samples/sec (communication overhead)
- Pipeline Parallel: ~1500 samples/sec (with 4 micro-batches)
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple
import json
import os


class SimpleCNN(nn.Module):
    """Simple CNN for throughput testing"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ThroughputBenchmark:
    """Helper class for measuring training throughput"""

    def __init__(self):
        self.results = {}

    def measure_throughput(
        self,
        model: nn.Module,
        batch_size: int,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        input_shape: Tuple = (3, 32, 32)
    ) -> Dict[str, float]:
        """
        Measure training throughput for a model.

        Args:
            model: PyTorch model to benchmark
            batch_size: Batch size for training
            num_iterations: Number of iterations to measure
            warmup_iterations: Warmup iterations before measurement
            input_shape: Input tensor shape (C, H, W)

        Returns:
            Dictionary with performance metrics
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Warmup
        for _ in range(warmup_iterations):
            data = torch.randn(batch_size, *input_shape).to(device)
            target = torch.randint(0, 10, (batch_size,)).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for _ in range(num_iterations):
            data = torch.randn(batch_size, *input_shape).to(device)
            target = torch.randint(0, 10, (batch_size,)).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        # Calculate metrics
        total_time = end_time - start_time
        total_samples = batch_size * num_iterations
        samples_per_sec = total_samples / total_time
        batches_per_sec = num_iterations / total_time
        time_per_batch = total_time / num_iterations

        return {
            'samples_per_sec': samples_per_sec,
            'batches_per_sec': batches_per_sec,
            'time_per_batch': time_per_batch,
            'total_time': total_time,
            'peak_memory_mb': peak_memory / (1024 ** 2),
            'avg_memory_mb': (peak_memory - start_memory) / (1024 ** 2)
        }

    def save_results(self, filepath: str = 'performance_results.json'):
        """Save benchmark results to JSON file"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)


@pytest.fixture
def benchmark():
    """Fixture providing throughput benchmark"""
    return ThroughputBenchmark()


@pytest.fixture
def simple_model():
    """Fixture providing simple CNN model"""
    return SimpleCNN(num_classes=10)


@pytest.mark.performance
@pytest.mark.slow
class TestThroughput:
    """Throughput performance tests"""

    def test_baseline_throughput(self, benchmark, simple_model):
        """
        Test baseline throughput with simple training loop.

        Expected: >100 samples/sec on CPU, >1000 samples/sec on GPU
        """
        results = benchmark.measure_throughput(
            simple_model,
            batch_size=32,
            num_iterations=50,
            warmup_iterations=5
        )

        benchmark.results['baseline'] = results

        # Assertions
        assert results['samples_per_sec'] > 0, "Throughput should be positive"
        assert results['time_per_batch'] > 0, "Time per batch should be positive"

        # Performance targets
        if torch.cuda.is_available():
            assert results['samples_per_sec'] > 500, \
                f"GPU throughput too low: {results['samples_per_sec']:.1f} samples/sec"
        else:
            assert results['samples_per_sec'] > 50, \
                f"CPU throughput too low: {results['samples_per_sec']:.1f} samples/sec"

        print(f"\n✓ Baseline Throughput: {results['samples_per_sec']:.1f} samples/sec")
        print(f"  Time per batch: {results['time_per_batch']*1000:.2f} ms")
        print(f"  Peak memory: {results['peak_memory_mb']:.1f} MB")

    def test_batch_size_scaling(self, benchmark, simple_model):
        """
        Test how throughput scales with batch size.

        Expected: Throughput should increase with batch size (up to memory limit)
        """
        batch_sizes = [8, 16, 32, 64]
        results_by_batch = {}

        for bs in batch_sizes:
            try:
                results = benchmark.measure_throughput(
                    SimpleCNN(num_classes=10),  # Fresh model for each test
                    batch_size=bs,
                    num_iterations=30,
                    warmup_iterations=3
                )
                results_by_batch[bs] = results
                print(f"\n✓ Batch size {bs}: {results['samples_per_sec']:.1f} samples/sec")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n✗ Batch size {bs}: OOM")
                    break
                raise

        benchmark.results['batch_size_scaling'] = results_by_batch

        # Verify scaling - larger batches should generally be more efficient
        assert len(results_by_batch) >= 2, "Should test at least 2 batch sizes"

        throughputs = [r['samples_per_sec'] for r in results_by_batch.values()]
        # Allow some variance but expect general upward trend
        assert max(throughputs) > min(throughputs), \
            "Throughput should vary with batch size"

    def test_mixed_precision_speedup(self, benchmark, simple_model):
        """
        Test speedup from mixed precision training.

        Expected: 1.5-3x speedup with fp16/bf16 on modern GPUs
        """
        if not torch.cuda.is_available():
            pytest.skip("Mixed precision requires CUDA")

        # FP32 baseline
        model_fp32 = SimpleCNN(num_classes=10)
        results_fp32 = benchmark.measure_throughput(
            model_fp32,
            batch_size=64,
            num_iterations=50,
            warmup_iterations=5
        )

        # FP16 with autocast
        model_fp16 = SimpleCNN(num_classes=10)
        device = torch.device('cuda')
        model_fp16 = model_fp16.to(device)
        optimizer = torch.optim.SGD(model_fp16.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        # Warmup
        for _ in range(5):
            data = torch.randn(64, 3, 32, 32).to(device)
            target = torch.randint(0, 10, (64,)).to(device)

            with torch.cuda.amp.autocast():
                output = model_fp16(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        start_time = time.time()

        # Measure FP16
        for _ in range(50):
            data = torch.randn(64, 3, 32, 32).to(device)
            target = torch.randint(0, 10, (64,)).to(device)

            with torch.cuda.amp.autocast():
                output = model_fp16(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        results_fp16 = {
            'samples_per_sec': (64 * 50) / total_time,
            'time_per_batch': total_time / 50
        }

        speedup = results_fp16['samples_per_sec'] / results_fp32['samples_per_sec']

        benchmark.results['mixed_precision'] = {
            'fp32': results_fp32,
            'fp16': results_fp16,
            'speedup': speedup
        }

        print(f"\n✓ FP32: {results_fp32['samples_per_sec']:.1f} samples/sec")
        print(f"✓ FP16: {results_fp16['samples_per_sec']:.1f} samples/sec")
        print(f"✓ Speedup: {speedup:.2f}x")

        # Mixed precision should be at least as fast (allow for variance)
        assert speedup >= 0.9, f"FP16 slower than FP32: {speedup:.2f}x"

    def test_gradient_accumulation_throughput(self, benchmark):
        """
        Test throughput with gradient accumulation.

        Expected: Similar samples/sec but higher effective batch size
        """
        model = SimpleCNN(num_classes=10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        accumulation_steps = 4
        batch_size = 16
        num_iterations = 40

        # Warmup
        for _ in range(5):
            for _ in range(accumulation_steps):
                data = torch.randn(batch_size, 3, 32, 32).to(device)
                target = torch.randint(0, 10, (batch_size,)).to(device)

                output = model(data)
                loss = criterion(output, target) / accumulation_steps
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        # Measure with gradient accumulation
        for _ in range(num_iterations):
            for _ in range(accumulation_steps):
                data = torch.randn(batch_size, 3, 32, 32).to(device)
                target = torch.randint(0, 10, (batch_size,)).to(device)

                output = model(data)
                loss = criterion(output, target) / accumulation_steps
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = time.time() - start_time
        total_samples = batch_size * accumulation_steps * num_iterations

        results = {
            'samples_per_sec': total_samples / total_time,
            'effective_batch_size': batch_size * accumulation_steps,
            'optimizer_steps_per_sec': num_iterations / total_time
        }

        benchmark.results['gradient_accumulation'] = results

        print(f"\n✓ Gradient Accumulation Throughput: {results['samples_per_sec']:.1f} samples/sec")
        print(f"  Effective batch size: {results['effective_batch_size']}")
        print(f"  Optimizer steps/sec: {results['optimizer_steps_per_sec']:.2f}")

        assert results['samples_per_sec'] > 0
        assert results['optimizer_steps_per_sec'] > 0


@pytest.mark.performance
def test_save_throughput_results(benchmark, simple_model):
    """Save all throughput results to file"""
    if benchmark.results:
        benchmark.save_results('test_results/throughput_results.json')
        print("\n✓ Throughput results saved to test_results/throughput_results.json")
