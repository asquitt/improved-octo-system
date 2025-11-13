"""
Comprehensive unit tests for model parallelism.

Tests tensor parallel layers, communication patterns, and memory efficiency.
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import os

sys.path.insert(0, 'src')

from distributed_training.model_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelModel,
    get_model_memory_usage,
)


class TestModelParallel:
    """Test suite for model parallelism."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    def test_column_parallel_linear_creation(self):
        """Test creation of column parallel linear layer."""
        layer = ColumnParallelLinear(
            in_features=128,
            out_features=256,
            bias=True,
            gather_output=False,
        )

        assert layer is not None
        assert layer.weight is not None

        # Check dimensions
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        expected_out = 256 // world_size
        assert layer.weight.shape[0] == expected_out
        assert layer.weight.shape[1] == 128

    def test_column_parallel_forward(self):
        """Test forward pass of column parallel layer."""
        batch_size = 4
        in_features = 128
        out_features = 256

        layer = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            gather_output=False,
        )

        # Create input
        x = torch.randn(batch_size, in_features)

        # Forward pass
        output = layer(x)

        # Check output shape
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        expected_out = out_features // world_size
        assert output.shape == (batch_size, expected_out)

    def test_row_parallel_linear_creation(self):
        """Test creation of row parallel linear layer."""
        layer = RowParallelLinear(
            in_features=256,
            out_features=128,
            bias=True,
            input_is_parallel=False,
        )

        assert layer is not None
        assert layer.weight is not None

        # Check dimensions
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        expected_in = 256 // world_size
        assert layer.weight.shape[0] == 128
        assert layer.weight.shape[1] == expected_in

    def test_row_parallel_forward(self):
        """Test forward pass of row parallel layer."""
        batch_size = 4
        in_features = 256
        out_features = 128

        layer = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            input_is_parallel=False,
        )

        # Create input
        x = torch.randn(batch_size, in_features)

        # Forward pass
        output = layer(x)

        # Check output shape
        assert output.shape == (batch_size, out_features)

    def test_tensor_parallel_model_creation(self):
        """Test creation of full tensor parallel model."""
        model = TensorParallelModel(
            input_size=128,
            hidden_size=256,
            output_size=10,
            num_layers=2,
        )

        assert model is not None
        assert len(model.layers) > 0

    def test_tensor_parallel_model_forward(self):
        """Test forward pass through tensor parallel model."""
        batch_size = 4
        input_size = 128
        output_size = 10

        model = TensorParallelModel(
            input_size=input_size,
            hidden_size=256,
            output_size=output_size,
            num_layers=2,
        )

        # Create input
        x = torch.randn(batch_size, input_size)

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (batch_size, output_size)

    def test_model_memory_usage(self):
        """Test memory usage calculation."""
        model = TensorParallelModel(
            input_size=128,
            hidden_size=256,
            output_size=10,
            num_layers=2,
        )

        memory_stats = get_model_memory_usage(model)

        assert "param_count" in memory_stats
        assert "total_size_mb" in memory_stats
        assert memory_stats["param_count"] > 0
        assert memory_stats["total_size_mb"] > 0

    def test_column_parallel_with_gather(self):
        """Test column parallel layer with output gathering."""
        batch_size = 4
        in_features = 128
        out_features = 256

        layer = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            gather_output=True,  # Gather output
        )

        x = torch.randn(batch_size, in_features)
        output = layer(x)

        # With gather, output should be full size
        # (but in single GPU mode, it's the same)
        assert output.shape[0] == batch_size

    def test_gradient_flow(self):
        """Test that gradients flow correctly through parallel layers."""
        model = TensorParallelModel(
            input_size=128,
            hidden_size=256,
            output_size=10,
            num_layers=2,
        )

        # Create input and target
        x = torch.randn(4, 128)
        target = torch.randint(0, 10, (4,))

        # Forward pass
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None

    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        model = TensorParallelModel(
            input_size=128,
            hidden_size=256,
            output_size=10,
            num_layers=2,
        )

        total_params = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters
        assert total_params > 1000  # At least some parameters
        assert total_params < 10_000_000  # Not unreasonably large


@pytest.mark.distributed
class TestDistributedModelParallel:
    """Tests that require actual distributed setup."""

    def test_column_parallel_distributed(self):
        """Test column parallel in actual distributed setting."""
        # This would need to be run with torchrun
        if not dist.is_initialized():
            pytest.skip("Requires distributed initialization")

        layer = ColumnParallelLinear(256, 512)
        x = torch.randn(4, 256).cuda()
        output = layer(x)

        # In distributed mode, output should be partitioned
        world_size = dist.get_world_size()
        assert output.shape[1] == 512 // world_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
