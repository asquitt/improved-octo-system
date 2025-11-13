"""
Example 4: Complete Training Demo

This demonstrates all features together:
- Data parallel training
- Checkpointing
- Fault tolerance
- Performance profiling
- Efficient data loading

Launch:
-------
torchrun --nproc_per_node=2 examples/04_complete_training_demo.py

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
from distributed_training.checkpointing import CheckpointManager
from distributed_training.profiling import PerformanceTracker
from distributed_training.utils import set_seed, is_main_process


class DemoModel(nn.Module):
    """Demo model for training."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


def main():
    """Complete training demo."""
    set_seed(42)

    if is_main_process():
        print("=" * 60)
        print("Complete Training Demo")
        print("=" * 60)
        print("\nThis demo showcases:")
        print("✓ Multi-GPU data parallel training")
        print("✓ Automatic checkpointing")
        print("✓ Performance profiling")
        print("✓ Efficient data loading")
        print("=" * 60 + "\n")

    # 1. Setup performance tracking
    tracker = PerformanceTracker()

    # 2. Create dataset
    with tracker.measure("dataset_creation"):
        print("Creating dataset...")
        num_samples = 2000
        X = torch.randn(num_samples, 28, 28)
        y = torch.randint(0, 10, (num_samples,))
        dataset = TensorDataset(X, y)

    # 3. Create optimized data loader
    with tracker.measure("dataloader_creation"):
        print("Creating distributed data loader...")
        train_loader = create_distributed_dataloader(
            dataset,
            batch_size=64,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    # 4. Create model
    print("Creating model...")
    model = DemoModel()

    # 5. Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="./checkpoints/complete_demo",
        max_checkpoints=3,
        best_metric="loss",
        best_mode="min",
    )

    # 6. Create trainer
    print("Creating DDP trainer...")
    trainer = DDPTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=nn.CrossEntropyLoss(),
        mixed_precision=True,
        gradient_accumulation_steps=2,
    )

    # 7. Training loop with profiling
    num_epochs = 5
    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        with tracker.measure(f"epoch_{epoch}"):
            epoch_loss = 0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                with tracker.measure("data_transfer"):
                    data = data.to(trainer.device)
                    target = target.to(trainer.device)

                with tracker.measure("forward"):
                    output = trainer.model(data)
                    loss = trainer.loss_fn(output, target)

                with tracker.measure("backward"):
                    loss.backward()

                with tracker.measure("optimizer_step"):
                    trainer.optimizer.step()
                    trainer.optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1
                tracker.record_samples(data.size(0))

                if batch_idx % 10 == 0 and is_main_process():
                    print(f"Epoch {epoch+1}/{num_epochs} "
                          f"[{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")

            # Calculate average loss
            avg_loss = epoch_loss / num_batches

            if is_main_process():
                print(f"\nEpoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}")

                # Save checkpoint
                checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=trainer.model,
                    optimizer=trainer.optimizer,
                    metrics={"loss": avg_loss},
                )

                # Print GPU stats
                gpu_stats = tracker.get_gpu_stats()
                print(f"GPU Memory: {gpu_stats.get('memory_allocated_gb', 0):.2f} GB\n")

    # 8. Print final statistics
    if is_main_process():
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        tracker.print_stats()

        # Save performance stats
        tracker.save_stats("./performance_stats.json")
        print("Performance stats saved to ./performance_stats.json")

    # 9. Cleanup
    trainer.cleanup()


if __name__ == "__main__":
    main()
