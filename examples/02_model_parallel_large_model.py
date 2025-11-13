"""
Example 2: Model Parallel Training (Large Model)

This example demonstrates model parallelism for models too large
to fit in single GPU memory. The model is split across GPUs.

Launch:
-------
torchrun --nproc_per_node=4 examples/02_model_parallel_large_model.py

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.insert(0, 'src')

from distributed_training.model_parallel import TensorParallelModel
from distributed_training.utils import set_seed, is_main_process


def main():
    """Main training function."""
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    set_seed(42)

    if is_main_process():
        print("=" * 60)
        print("Example 2: Model Parallel Training")
        print("=" * 60)
        print(f"\nUsing {world_size} GPUs for model parallelism")
        print("Model will be split across GPUs\n")

    # 1. Create large model (split across GPUs)
    print(f"[Rank {rank}] Creating tensor parallel model...")
    model = TensorParallelModel(
        input_size=1024,
        hidden_size=4096,  # Large hidden size
        output_size=1000,
        num_layers=4,
    ).cuda()

    # 2. Create dataset
    print(f"[Rank {rank}] Creating dataset...")
    num_samples = 500
    X = torch.randn(num_samples, 1024)
    y = torch.randint(0, 1000, (num_samples,))
    dataset = TensorDataset(X, y)

    # 3. Create data loader (note: all ranks get same data for model parallel)
    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
    )

    # 4. Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 5. Train
    if is_main_process():
        print("[Step] Starting training...")
        print("Each layer is split across GPUs (tensor parallelism)\n")

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()

            # Forward pass (automatic tensor parallel)
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0 and is_main_process():
                print(f"Epoch {epoch+1}/{num_epochs} "
                      f"[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        if is_main_process():
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}\n")

    if is_main_process():
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
