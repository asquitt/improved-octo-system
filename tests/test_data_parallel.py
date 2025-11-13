"""
Tests for data parallel training.

Run with: pytest tests/test_data_parallel.py
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.insert(0, 'src')

from distributed_training.data_parallel import DDPTrainer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    def forward(self, x):
        return self.layers(x)


def test_ddp_trainer_creation():
    """Test DDPTrainer initialization."""
    model = SimpleModel()

    trainer = DDPTrainer(
        model=model,
        num_gpus=1,
        checkpoint_dir="./test_checkpoints",
    )

    assert trainer is not None
    assert trainer.model is not None


def test_model_forward():
    """Test model forward pass."""
    model = SimpleModel()
    x = torch.randn(4, 10)
    output = model(x)

    assert output.shape == (4, 5)


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    from distributed_training.checkpointing import CheckpointManager

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters())

    manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints",
        max_checkpoints=2,
    )

    # Save checkpoint
    path = manager.save_checkpoint(
        epoch=0,
        model=model,
        optimizer=optimizer,
        metrics={"loss": 0.5},
    )

    # Load checkpoint
    if path:
        state = manager.load_checkpoint(path, model, optimizer)
        assert state["epoch"] == 0
        assert state["metrics"]["loss"] == 0.5


if __name__ == "__main__":
    test_ddp_trainer_creation()
    test_model_forward()
    test_checkpoint_save_load()
    print("All tests passed!")
