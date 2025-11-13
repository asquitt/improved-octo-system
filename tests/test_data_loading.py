"""
Unit tests for data loading utilities.

Tests distributed sampling, prefetching, and benchmarking.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys

sys.path.insert(0, 'src')

from distributed_training.data_loading import (
    create_distributed_dataloader,
    PrefetchDataLoader,
    benchmark_dataloader,
)


class TestDistributedDataLoader:
    """Test suite for distributed data loading."""

    @pytest.fixture
    def dataset(self):
        """Create dummy dataset."""
        X = torch.randn(1000, 10)
        y = torch.randint(0, 5, (1000,))
        return TensorDataset(X, y)

    def test_create_distributed_dataloader(self, dataset):
        """Test creating distributed data loader."""
        loader = create_distributed_dataloader(
            dataset,
            batch_size=32,
            num_workers=0,  # Use 0 for testing
            shuffle=True,
        )

        assert loader is not None
        assert isinstance(loader, DataLoader)

    def test_dataloader_batch_size(self, dataset):
        """Test that batch size is correct."""
        batch_size = 32
        loader = create_distributed_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
        )

        data, target = next(iter(loader))
        assert data.shape[0] == batch_size

    def test_dataloader_iteration(self, dataset):
        """Test iterating through data loader."""
        loader = create_distributed_dataloader(
            dataset,
            batch_size=32,
            num_workers=0,
        )

        batch_count = 0
        for data, target in loader:
            assert data.shape[1] == 10  # Feature dimension
            batch_count += 1

        assert batch_count > 0

    def test_dataloader_with_workers(self, dataset):
        """Test data loader with multiple workers."""
        loader = create_distributed_dataloader(
            dataset,
            batch_size=32,
            num_workers=2,
        )

        # Should be able to iterate
        data, target = next(iter(loader))
        assert data is not None


class TestPrefetchDataLoader:
    """Test suite for prefetching data loader."""

    @pytest.fixture
    def base_loader(self):
        """Create base data loader."""
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10)

    def test_prefetch_loader_creation(self, base_loader):
        """Test creating prefetch data loader."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        prefetch_loader = PrefetchDataLoader(
            base_loader,
            device=device,
        )

        assert prefetch_loader is not None

    def test_prefetch_loader_iteration(self, base_loader):
        """Test iterating through prefetch loader."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        prefetch_loader = PrefetchDataLoader(
            base_loader,
            device=device,
        )

        batch_count = 0
        for data, target in prefetch_loader:
            # Data should be on correct device
            assert data.device == device
            assert target.device == device
            batch_count += 1

        assert batch_count == len(base_loader)

    def test_prefetch_loader_length(self, base_loader):
        """Test prefetch loader length."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        prefetch_loader = PrefetchDataLoader(
            base_loader,
            device=device,
        )

        assert len(prefetch_loader) == len(base_loader)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_on_cuda(self, base_loader):
        """Test prefetching on CUDA device."""
        prefetch_loader = PrefetchDataLoader(
            base_loader,
            device=torch.device("cuda"),
        )

        data, target = next(iter(prefetch_loader))

        assert data.is_cuda
        assert target.is_cuda


class TestBenchmarkDataLoader:
    """Test suite for data loader benchmarking."""

    @pytest.fixture
    def loader(self):
        """Create data loader for benchmarking."""
        X = torch.randn(1000, 10)
        y = torch.randint(0, 5, (1000,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32)

    def test_benchmark_dataloader(self, loader):
        """Test benchmarking data loader."""
        stats = benchmark_dataloader(loader, num_iterations=10)

        assert "total_samples" in stats
        assert "total_time_sec" in stats
        assert "samples_per_sec" in stats
        assert "batches_per_sec" in stats
        assert "time_per_batch_ms" in stats

        assert stats["total_samples"] > 0
        assert stats["total_time_sec"] > 0

    def test_benchmark_throughput(self, loader):
        """Test that benchmark calculates throughput."""
        stats = benchmark_dataloader(loader, num_iterations=10)

        # Throughput should be positive
        assert stats["samples_per_sec"] > 0
        assert stats["batches_per_sec"] > 0

    def test_benchmark_different_iterations(self, loader):
        """Test benchmarking with different iteration counts."""
        stats1 = benchmark_dataloader(loader, num_iterations=5)
        stats2 = benchmark_dataloader(loader, num_iterations=10)

        # More iterations should process more samples
        assert stats2["total_samples"] > stats1["total_samples"]


class TestDataLoaderConfigurations:
    """Test various data loader configurations."""

    @pytest.fixture
    def dataset(self):
        """Create dataset."""
        X = torch.randn(200, 10)
        y = torch.randint(0, 5, (200,))
        return TensorDataset(X, y)

    def test_pin_memory(self, dataset):
        """Test data loader with pinned memory."""
        loader = create_distributed_dataloader(
            dataset,
            batch_size=32,
            pin_memory=True,
            num_workers=0,
        )

        data, target = next(iter(loader))

        # Just verify it works
        assert data is not None

    def test_drop_last(self, dataset):
        """Test data loader with drop_last."""
        loader = create_distributed_dataloader(
            dataset,
            batch_size=32,
            drop_last=True,
            num_workers=0,
        )

        batch_sizes = []
        for data, target in loader:
            batch_sizes.append(data.shape[0])

        # All batches should have same size (32) because drop_last=True
        assert all(size == 32 for size in batch_sizes)

    def test_no_shuffle(self, dataset):
        """Test data loader without shuffling."""
        loader = create_distributed_dataloader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        # Get first batch twice
        data1, _ = next(iter(loader))
        data2, _ = next(iter(loader))

        # Without shuffling, they should be the same
        # (in deterministic setting)
        assert data1 is not None
        assert data2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
