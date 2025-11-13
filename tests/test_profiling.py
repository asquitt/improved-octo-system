"""
Unit tests for performance profiling.

Tests performance tracking, GPU stats, and benchmarking.
"""

import pytest
import torch
import torch.nn as nn
import time
import sys

sys.path.insert(0, 'src')

from distributed_training.profiling import (
    PerformanceTracker,
    profile_model,
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        return self.layers(x)


class TestPerformanceTracker:
    """Test suite for performance tracking."""

    @pytest.fixture
    def tracker(self):
        """Create performance tracker."""
        return PerformanceTracker()

    def test_tracker_creation(self, tracker):
        """Test tracker initialization."""
        assert tracker is not None
        assert len(tracker.timers) == 0

    def test_start_stop_timer(self, tracker):
        """Test starting and stopping timers."""
        tracker.start_timer("test_operation")
        time.sleep(0.1)  # Simulate work
        tracker.stop_timer("test_operation")

        assert "test_operation" in tracker.timers
        assert len(tracker.timers["test_operation"]) == 1
        assert tracker.timers["test_operation"][0] >= 0.1

    def test_measure_context_manager(self, tracker):
        """Test measure context manager."""
        with tracker.measure("test_context"):
            time.sleep(0.05)

        assert "test_context" in tracker.timers
        assert tracker.timers["test_context"][0] >= 0.05

    def test_multiple_measurements(self, tracker):
        """Test multiple measurements of same operation."""
        for _ in range(5):
            with tracker.measure("operation"):
                time.sleep(0.01)

        assert len(tracker.timers["operation"]) == 5

    def test_record_samples(self, tracker):
        """Test recording sample counts."""
        tracker.record_samples(32)
        tracker.record_samples(32)
        tracker.record_samples(32)

        assert len(tracker.sample_counts) == 3
        assert sum(tracker.sample_counts) == 96

    def test_get_stats(self, tracker):
        """Test getting statistics."""
        # Record some measurements
        for _ in range(10):
            with tracker.measure("forward"):
                time.sleep(0.01)

        tracker.record_samples(320)

        stats = tracker.get_stats()

        assert "forward" in stats
        assert "avg_ms" in stats["forward"]
        assert "min_ms" in stats["forward"]
        assert "max_ms" in stats["forward"]
        assert "count" in stats["forward"]

        assert stats["forward"]["count"] == 10
        assert stats["forward"]["avg_ms"] >= 10  # At least 10ms

    def test_throughput_calculation(self, tracker):
        """Test throughput calculation."""
        # Simulate processing batches
        for _ in range(10):
            tracker.record_samples(32)
            time.sleep(0.01)

        stats = tracker.get_stats()

        assert "throughput" in stats
        assert "samples_per_sec" in stats["throughput"]
        assert stats["throughput"]["total_samples"] == 320

    def test_gpu_stats(self, tracker):
        """Test GPU statistics collection."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        gpu_stats = tracker.get_gpu_stats()

        assert "memory_allocated_gb" in gpu_stats
        assert "memory_reserved_gb" in gpu_stats
        assert gpu_stats["memory_allocated_gb"] >= 0

    def test_reset(self, tracker):
        """Test resetting tracker."""
        with tracker.measure("operation"):
            time.sleep(0.01)

        tracker.record_samples(32)

        tracker.reset()

        assert len(tracker.timers) == 0
        assert len(tracker.sample_counts) == 0

    def test_print_stats(self, tracker, capsys):
        """Test printing statistics."""
        with tracker.measure("test"):
            time.sleep(0.01)

        tracker.print_stats()

        captured = capsys.readouterr()
        assert "Performance Statistics" in captured.out
        assert "test" in captured.out

    def test_save_stats(self, tracker, tmp_path):
        """Test saving statistics to file."""
        with tracker.measure("operation"):
            time.sleep(0.01)

        filepath = tmp_path / "stats.json"
        tracker.save_stats(str(filepath))

        assert filepath.exists()

        import json
        with open(filepath) as f:
            stats = json.load(f)

        assert "timers" in stats
        assert "gpu" in stats


class TestProfileModel:
    """Test suite for model profiling."""

    def test_profile_simple_model(self):
        """Test profiling a simple model."""
        model = DummyModel()

        stats = profile_model(
            model=model,
            input_shape=(100,),
            device="cpu",
        )

        assert "total_params" in stats
        assert "trainable_params" in stats
        assert "param_memory_mb" in stats
        assert "inference_time_ms" in stats
        assert "throughput_samples_per_sec" in stats

        assert stats["total_params"] > 0
        assert stats["inference_time_ms"] > 0

    def test_profile_on_cuda(self):
        """Test profiling on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = DummyModel()

        stats = profile_model(
            model=model,
            input_shape=(100,),
            device="cuda",
        )

        assert stats["total_params"] > 0

    def test_parameter_counting(self):
        """Test that parameter counting is accurate."""
        model = DummyModel()

        stats = profile_model(
            model=model,
            input_shape=(100,),
            device="cpu",
        )

        # Manual count
        expected_params = sum(p.numel() for p in model.parameters())

        assert stats["total_params"] == expected_params

    def test_memory_calculation(self):
        """Test memory calculation."""
        model = DummyModel()

        stats = profile_model(
            model=model,
            input_shape=(100,),
            device="cpu",
        )

        # Memory should be positive
        assert stats["param_memory_mb"] > 0

    def test_throughput_measurement(self):
        """Test throughput measurement."""
        model = DummyModel()

        stats = profile_model(
            model=model,
            input_shape=(100,),
            device="cpu",
        )

        # Throughput should be positive
        assert stats["throughput_samples_per_sec"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
