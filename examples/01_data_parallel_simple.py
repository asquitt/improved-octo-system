"""
Example 1: Data Parallel Training (Simple)

This example demonstrates basic multi-GPU training using PyTorch DDP.
Perfect for models that fit in single GPU memory.

Launch:
-------
torchrun --nproc_per_node=4 examples/01_data_parallel_simple.py

or for single node:
torchrun --nproc_per_node=2 examples/01_data_parallel_simple.py

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import sys
sys.path.insert(0, 'src')

from distributed_training.data_parallel import DDPTrainer
from distributed_training.data_loading import create_distributed_dataloader
from distributed_training.utils import set_seed


# 1. Define a simple model
class SimpleModel(nn.Module):
    """Simple CNN for demonstration."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    """Main training function."""
    # Set seed for reproducibility
    set_seed(42)

    print("=" * 60)
    print("Example 1: Data Parallel Training")
    print("=" * 60)

    # 2. Create synthetic dataset (replace with real data)
    print("\n[Step 1] Creating dataset...")
    num_samples = 1000
    X = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)

    # 3. Create distributed data loader
    print("[Step 2] Creating distributed data loader...")
    train_loader = create_distributed_dataloader(
        dataset,
        batch_size=32,
        num_workers=2,
        shuffle=True,
    )

    # 4. Create model
    print("[Step 3] Creating model...")
    model = SimpleModel()

    # 5. Create DDP trainer
    print("[Step 4] Initializing DDP trainer...")
    trainer = DDPTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        num_gpus=torch.cuda.device_count(),
        mixed_precision=True,  # Use AMP for faster training
        gradient_accumulation_steps=2,
        checkpoint_dir="./checkpoints/data_parallel",
    )

    # 6. Train!
    print("[Step 5] Starting training...")
    print("This will train the model across all available GPUs")
    print("Each GPU processes different batches (data parallelism)\n")

    trainer.train(
        train_loader=train_loader,
        num_epochs=5,
        log_interval=5,
        save_interval=2,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # 7. Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()
