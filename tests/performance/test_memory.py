"""
Memory Performance Tests

Tests that measure memory usage and efficiency for different strategies.
These tests help identify memory bottlenecks and verify memory optimizations.

Memory Metrics:
- Peak memory usage
- Memory per parameter
- Activation memory
- Optimizer state memory

Expected Baselines (for 100M parameter model):
- DDP: ~1.6 GB (2 bytes per param for fp16 + optimizer state)
- FSDP: ~800 MB (sharded across ranks)
- DeepSpeed ZeRO-3: ~400 MB (full sharding)
- Activation Checkpointing: 50-80% memory reduction
"""

import pytest
import torch
import torch.nn as nn
import gc
from typing import Dict, Tuple
import json


class LargeModel(nn.Module):
    """Large model for memory testing"""
    def __init__(self, hidden_size=1024, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return x


class MemoryTracker:
    """Helper class for tracking memory usage"""

    def __init__(self):
        self.results = {}

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {'allocated_mb': 0, 'reserved_mb': 0, 'peak_mb': 0}

        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
            'reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
            'peak_mb': torch.cuda.max_memory_allocated() / (1024 ** 2)
        }

    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def measure_model_memory(
        self,
        model: nn.Module,
        input_shape: Tuple,
        batch_size: int = 32,
        with_backward: bool = True
    ) -> Dict[str, float]:
        """
        Measure memory usage for a model.

        Args:
            model: PyTorch model to measure
            input_shape: Input tensor shape
            batch_size: Batch size for forward pass
            with_backward: Whether to include backward pass

        Returns:
            Dictionary with memory metrics
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Measure model parameters
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

        initial_stats = self.get_memory_stats()

        # Forward pass
        x = torch.randn(batch_size, *input_shape).to(device)
        if with_backward:
            x.requires_grad = True

        output = model(x)

        forward_stats = self.get_memory_stats()

        # Backward pass
        if with_backward:
            loss = output.sum()
            loss.backward()

        final_stats = self.get_memory_stats()

        return {
            'param_memory_mb': param_memory,
            'forward_memory_mb': forward_stats['peak_mb'] - initial_stats['allocated_mb'],
            'backward_memory_mb': final_stats['peak_mb'] - forward_stats['peak_mb'],
            'total_peak_mb': final_stats['peak_mb'],
            'activation_memory_mb': forward_stats['allocated_mb'] - param_memory
        }

    def save_results(self, filepath: str = 'memory_results.json'):
        """Save memory results to JSON file"""
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)


@pytest.fixture
def memory_tracker():
    """Fixture providing memory tracker"""
    return MemoryTracker()


@pytest.fixture
def large_model():
    """Fixture providing large model"""
    return LargeModel(hidden_size=1024, num_layers=6)


@pytest.mark.performance
@pytest.mark.slow
class TestMemory:
    """Memory performance tests"""

    def test_baseline_memory_usage(self, memory_tracker, large_model):
        """
        Test baseline memory usage for model training.

        Expected: Memory scales with model size and batch size
        """
        results = memory_tracker.measure_model_memory(
            large_model,
            input_shape=(1024,),
            batch_size=32,
            with_backward=True
        )

        memory_tracker.results['baseline'] = results

        # Assertions
        assert results['param_memory_mb'] > 0, "Parameter memory should be positive"
        assert results['total_peak_mb'] > results['param_memory_mb'], \
            "Total memory should exceed parameter memory"

        print(f"\n✓ Baseline Memory Usage:")
        print(f"  Parameters: {results['param_memory_mb']:.1f} MB")
        print(f"  Forward pass: {results['forward_memory_mb']:.1f} MB")
        print(f"  Backward pass: {results['backward_memory_mb']:.1f} MB")
        print(f"  Total peak: {results['total_peak_mb']:.1f} MB")

    def test_batch_size_memory_scaling(self, memory_tracker):
        """
        Test how memory scales with batch size.

        Expected: Linear scaling of activation memory with batch size
        """
        model = LargeModel(hidden_size=512, num_layers=4)
        batch_sizes = [8, 16, 32, 64]
        results_by_batch = {}

        for bs in batch_sizes:
            try:
                results = memory_tracker.measure_model_memory(
                    LargeModel(hidden_size=512, num_layers=4),
                    input_shape=(512,),
                    batch_size=bs,
                    with_backward=True
                )
                results_by_batch[bs] = results
                print(f"\n✓ Batch size {bs}: {results['total_peak_mb']:.1f} MB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n✗ Batch size {bs}: OOM")
                    break
                raise

        memory_tracker.results['batch_size_scaling'] = results_by_batch

        # Verify memory increases with batch size
        assert len(results_by_batch) >= 2, "Should test at least 2 batch sizes"

        memories = [r['total_peak_mb'] for r in results_by_batch.values()]
        assert memories[-1] > memories[0], \
            "Memory should increase with batch size"

    def test_gradient_checkpointing_memory_savings(self, memory_tracker):
        """
        Test memory savings from gradient checkpointing.

        Expected: 50-80% reduction in activation memory
        """
        if not torch.cuda.is_available():
            pytest.skip("Gradient checkpointing test requires CUDA")

        # Model without checkpointing
        model_no_ckpt = LargeModel(hidden_size=512, num_layers=8)
        results_no_ckpt = memory_tracker.measure_model_memory(
            model_no_ckpt,
            input_shape=(512,),
            batch_size=32,
            with_backward=True
        )

        # Model with checkpointing
        from torch.utils.checkpoint import checkpoint_sequential

        class CheckpointedModel(nn.Module):
            def __init__(self, hidden_size=512, num_layers=8):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
                ])
                self.relu = nn.ReLU()

            def forward(self, x):
                # Use gradient checkpointing for layers
                x = checkpoint_sequential(self.layers, 2, x)
                return x

        model_ckpt = CheckpointedModel(hidden_size=512, num_layers=8)
        results_ckpt = memory_tracker.measure_model_memory(
            model_ckpt,
            input_shape=(512,),
            batch_size=32,
            with_backward=True
        )

        memory_savings = 1 - (results_ckpt['total_peak_mb'] / results_no_ckpt['total_peak_mb'])

        memory_tracker.results['gradient_checkpointing'] = {
            'without_checkpoint': results_no_ckpt,
            'with_checkpoint': results_ckpt,
            'memory_savings': memory_savings
        }

        print(f"\n✓ Memory Usage:")
        print(f"  Without checkpointing: {results_no_ckpt['total_peak_mb']:.1f} MB")
        print(f"  With checkpointing: {results_ckpt['total_peak_mb']:.1f} MB")
        print(f"  Savings: {memory_savings*100:.1f}%")

        # Should see some memory savings
        assert memory_savings > 0, "Checkpointing should reduce memory"

    def test_optimizer_state_memory(self, memory_tracker, large_model):
        """
        Test memory usage of different optimizers.

        Expected: Adam uses 2x more memory than SGD (momentum + variance)
        """
        if not torch.cuda.is_available():
            pytest.skip("Optimizer memory test requires CUDA")

        device = torch.device('cuda')

        # Test SGD
        model_sgd = LargeModel(hidden_size=512, num_layers=4).to(device)
        optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)

        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(32, 512).to(device)
        loss = model_sgd(x).sum()
        loss.backward()
        optimizer_sgd.step()
        sgd_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Test Adam
        model_adam = LargeModel(hidden_size=512, num_layers=4).to(device)
        optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)

        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(32, 512).to(device)
        loss = model_adam(x).sum()
        loss.backward()
        optimizer_adam.step()
        adam_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

        memory_tracker.results['optimizer_memory'] = {
            'sgd_mb': sgd_memory,
            'adam_mb': adam_memory,
            'adam_overhead': (adam_memory / sgd_memory) - 1
        }

        print(f"\n✓ Optimizer Memory:")
        print(f"  SGD: {sgd_memory:.1f} MB")
        print(f"  Adam: {adam_memory:.1f} MB")
        print(f"  Adam overhead: {((adam_memory / sgd_memory) - 1)*100:.1f}%")

        # Adam should use more memory than SGD
        assert adam_memory > sgd_memory, "Adam should use more memory than SGD"

    def test_mixed_precision_memory_savings(self, memory_tracker):
        """
        Test memory savings from mixed precision training.

        Expected: ~2x reduction in model parameter memory
        """
        if not torch.cuda.is_available():
            pytest.skip("Mixed precision test requires CUDA")

        # FP32 model
        model_fp32 = LargeModel(hidden_size=1024, num_layers=6)
        results_fp32 = memory_tracker.measure_model_memory(
            model_fp32,
            input_shape=(1024,),
            batch_size=32,
            with_backward=True
        )

        # FP16 model
        model_fp16 = LargeModel(hidden_size=1024, num_layers=6).half()
        device = torch.device('cuda')
        model_fp16 = model_fp16.to(device)

        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(32, 1024).half().to(device)
        x.requires_grad = True
        output = model_fp16(x)
        loss = output.sum()
        loss.backward()

        results_fp16 = {
            'param_memory_mb': sum(p.numel() * p.element_size() for p in model_fp16.parameters()) / (1024 ** 2),
            'total_peak_mb': torch.cuda.max_memory_allocated() / (1024 ** 2)
        }

        param_savings = 1 - (results_fp16['param_memory_mb'] / results_fp32['param_memory_mb'])

        memory_tracker.results['mixed_precision'] = {
            'fp32': results_fp32,
            'fp16': results_fp16,
            'param_memory_savings': param_savings
        }

        print(f"\n✓ Mixed Precision Memory:")
        print(f"  FP32 parameters: {results_fp32['param_memory_mb']:.1f} MB")
        print(f"  FP16 parameters: {results_fp16['param_memory_mb']:.1f} MB")
        print(f"  Savings: {param_savings*100:.1f}%")

        # Should see ~2x savings in parameter memory
        assert param_savings > 0.4, f"Expected >40% savings, got {param_savings*100:.1f}%"


@pytest.mark.performance
def test_save_memory_results(memory_tracker):
    """Save all memory results to file"""
    if memory_tracker.results:
        memory_tracker.save_results('test_results/memory_results.json')
        print("\n✓ Memory results saved to test_results/memory_results.json")
