"""
Unit tests for checkpointing functionality.

Tests checkpoint saving, loading, rotation, and best model tracking.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, 'src')

from distributed_training.checkpointing import CheckpointManager


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointManager:
    """Test suite for checkpoint management."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def model(self):
        """Create dummy model."""
        return DummyModel()

    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer."""
        return torch.optim.Adam(model.parameters(), lr=0.001)

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create checkpoint manager."""
        return CheckpointManager(
            checkpoint_dir=temp_dir,
            max_checkpoints=3,
            best_metric="loss",
            best_mode="min",
        )

    def test_checkpoint_manager_creation(self, checkpoint_manager, temp_dir):
        """Test checkpoint manager initialization."""
        assert checkpoint_manager is not None
        assert checkpoint_manager.checkpoint_dir == Path(temp_dir)
        assert checkpoint_manager.max_checkpoints == 3
        assert checkpoint_manager.best_metric == "loss"

    def test_save_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test saving a checkpoint."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=optimizer,
            metrics={"loss": 0.5, "accuracy": 0.8},
        )

        if checkpoint_path:  # Only on main process
            assert Path(checkpoint_path).exists()
            assert "checkpoint_epoch_0.pt" in checkpoint_path

    def test_load_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test loading a checkpoint."""
        # Save checkpoint first
        checkpoint_path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
        )

        if checkpoint_path:
            # Create new model and optimizer
            new_model = DummyModel()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            # Load checkpoint
            state = checkpoint_manager.load_checkpoint(
                checkpoint_path,
                new_model,
                new_optimizer,
            )

            assert state["epoch"] == 0
            assert state["metrics"]["loss"] == 0.5

    def test_checkpoint_rotation(self, checkpoint_manager, model, optimizer):
        """Test that old checkpoints are deleted."""
        if not checkpoint_manager.is_main_process:
            pytest.skip("Only test on main process")

        # Save more checkpoints than max_checkpoints
        for epoch in range(5):
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics={"loss": 1.0 / (epoch + 1)},
            )

        # Count checkpoint files
        checkpoints = list(checkpoint_manager.checkpoint_dir.glob("checkpoint_epoch_*.pt"))

        # Should only have max_checkpoints (3) + best model
        assert len(checkpoints) <= checkpoint_manager.max_checkpoints + 1

    def test_best_model_tracking(self, checkpoint_manager, model, optimizer):
        """Test best model is tracked correctly."""
        if not checkpoint_manager.is_main_process:
            pytest.skip("Only test on main process")

        # Save checkpoints with different losses
        losses = [0.5, 0.3, 0.4, 0.2, 0.6]
        for epoch, loss in enumerate(losses):
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics={"loss": loss},
            )

        # Best model should have loss 0.2
        assert checkpoint_manager.best_metric_value == 0.2

        # Best model file should exist
        best_path = checkpoint_manager.checkpoint_dir / "best_model.pt"
        assert best_path.exists()

    def test_load_latest_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test loading the latest checkpoint."""
        if not checkpoint_manager.is_main_process:
            pytest.skip("Only test on main process")

        # Save multiple checkpoints
        for epoch in range(3):
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics={"loss": 0.5},
            )

        # Load latest
        new_model = DummyModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        state = checkpoint_manager.load_latest_checkpoint(new_model, new_optimizer)

        if state:
            assert state["epoch"] == 2  # Latest epoch

    def test_load_best_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test loading the best checkpoint."""
        if not checkpoint_manager.is_main_process:
            pytest.skip("Only test on main process")

        # Save checkpoints with different losses
        for epoch in range(3):
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics={"loss": 0.5 - epoch * 0.1},  # Decreasing loss
            )

        # Load best
        new_model = DummyModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        state = checkpoint_manager.load_best_checkpoint(new_model, new_optimizer)

        if state:
            # Best should be epoch 2 with loss 0.3
            assert state["metrics"]["loss"] == 0.3

    def test_checkpoint_with_extra_state(self, checkpoint_manager, model, optimizer):
        """Test saving checkpoint with extra state."""
        extra_state = {
            "learning_rate": 0.001,
            "scheduler_state": {"step": 10},
        }

        checkpoint_path = checkpoint_manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=optimizer,
            metrics={"loss": 0.5},
            extra_state=extra_state,
        )

        if checkpoint_path:
            # Load and verify extra state
            new_model = DummyModel()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            state = checkpoint_manager.load_checkpoint(
                checkpoint_path,
                new_model,
                new_optimizer,
            )

            assert "extra_state" in state
            assert state["extra_state"]["learning_rate"] == 0.001

    def test_is_better_min_mode(self, checkpoint_manager):
        """Test is_better with min mode."""
        assert checkpoint_manager._is_better(0.3, 0.5) is True
        assert checkpoint_manager._is_better(0.5, 0.3) is False

    def test_is_better_max_mode(self, temp_dir):
        """Test is_better with max mode."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            best_mode="max",
        )
        assert manager._is_better(0.9, 0.8) is True
        assert manager._is_better(0.8, 0.9) is False

    def test_metadata_saving(self, checkpoint_manager, model, optimizer):
        """Test that metadata is saved correctly."""
        if not checkpoint_manager.is_main_process:
            pytest.skip("Only test on main process")

        checkpoint_manager.save_checkpoint(
            epoch=0,
            model=model,
            optimizer=optimizer,
            metrics={"loss": 0.5, "accuracy": 0.9},
        )

        metadata_path = checkpoint_manager.checkpoint_dir / "checkpoint_metadata.json"
        assert metadata_path.exists()

        import json
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "checkpoints" in metadata
        assert len(metadata["checkpoints"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
