"""
Example 3: Pipeline Parallel Training (Transformer)

This example demonstrates pipeline parallelism where model layers
are partitioned across GPUs and micro-batches are pipelined.

Launch:
-------
torchrun --nproc_per_node=4 examples/03_pipeline_parallel_transformer.py

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
sys.path.insert(0, 'src')

from distributed_training.pipeline_parallel import (
    PipelineTrainer,
    calculate_pipeline_efficiency,
)
from distributed_training.utils import set_seed, is_main_process


# Simple transformer-like model for demonstration
class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


def create_transformer_model(num_layers=8, hidden_size=512, num_heads=8):
    """Create a sequential transformer model."""
    layers = []

    # Input embedding
    layers.append(nn.Linear(512, hidden_size))

    # Transformer blocks
    for _ in range(num_layers):
        layers.append(TransformerBlock(hidden_size, num_heads))

    # Output projection
    layers.append(nn.Linear(hidden_size, 1000))

    return nn.Sequential(*layers)


def main():
    """Main training function."""
    set_seed(42)

    if is_main_process():
        print("=" * 60)
        print("Example 3: Pipeline Parallel Training")
        print("=" * 60)

    # 1. Create transformer model
    print(f"Creating transformer model...")
    model = create_transformer_model(
        num_layers=8,
        hidden_size=512,
        num_heads=8,
    )

    # 2. Calculate pipeline efficiency
    num_stages = 4
    num_micro_batches = 8

    if is_main_process():
        efficiency = calculate_pipeline_efficiency(num_stages, num_micro_batches)
        print(f"\nPipeline Configuration:")
        print(f"  Stages: {num_stages}")
        print(f"  Micro-batches: {num_micro_batches}")
        print(f"  Expected Efficiency: {efficiency:.1f}%\n")

    # 3. Create dataset
    print("Creating dataset...")
    num_samples = 500
    X = torch.randn(num_samples, 32, 512)  # [batch, seq_len, features]
    y = torch.randint(0, 1000, (num_samples, 32))
    dataset = TensorDataset(X, y)

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 4. Create pipeline trainer
    print("Creating pipeline trainer...")
    trainer = PipelineTrainer(
        model=model,
        num_stages=num_stages,
        micro_batch_size=4,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    )

    # 5. Train
    if is_main_process():
        print("\nStarting pipeline training...")
        print("Model layers are split across GPUs")
        print("Micro-batches flow through the pipeline\n")

    trainer.train(
        train_loader=train_loader,
        num_epochs=5,
        num_micro_batches=num_micro_batches,
        log_interval=5,
    )

    if is_main_process():
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
