# Data Parallelism with PyTorch DDP

## Overview

Data Parallelism is the most common approach to distributed training. It works by:
1. **Replicating** the model on each GPU
2. **Splitting** the training data across GPUs
3. **Synchronizing** gradients after each backward pass

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Training Data                         │
│              [Batch Size: 128]                           │
└───────────┬─────────────────────────────┬───────────────┘
            │                             │
    ┌───────▼───────┐             ┌───────▼───────┐
    │   GPU 0       │             │   GPU 1       │
    │  Batch: 32    │             │  Batch: 32    │
    │  ┌─────────┐  │             │  ┌─────────┐  │
    │  │ Model   │  │             │  │ Model   │  │
    │  │ Replica │  │             │  │ Replica │  │
    │  └────┬────┘  │             │  └────┬────┘  │
    │       │       │             │       │       │
    │   Forward     │             │   Forward     │
    │   Backward    │             │   Backward    │
    │       │       │             │       │       │
    │   Gradients   │             │   Gradients   │
    └───────┬───────┘             └───────┬───────┘
            │                             │
            └──────────┬──────────────────┘
                       │
                   AllReduce
              (Average Gradients)
                       │
            ┌──────────▼──────────────┐
            │   Update Weights        │
            │   (Synchronized)        │
            └─────────────────────────┘
```

## Key Concepts

### 1. Process Group

A process group is a collection of processes that can communicate with each other. In DDP:
- Each GPU runs a separate process
- Processes communicate via NCCL (NVIDIA Collective Communications Library)
- All processes in a group can perform collective operations (like AllReduce)

```python
import torch.distributed as dist

# Initialize process group
dist.init_process_group(
    backend='nccl',          # Use NCCL for GPU communication
    init_method='env://',    # Read config from environment variables
    world_size=4,            # Total number of processes
    rank=0                   # This process's ID (0-3)
)
```

### 2. Rank and World Size

- **Rank**: Unique ID for each process (0 to world_size-1)
- **World Size**: Total number of processes (usually = number of GPUs)
- **Local Rank**: GPU ID on the current node (for multi-node training)

```python
rank = dist.get_rank()           # My process ID
world_size = dist.get_world_size()  # Total processes
```

### 3. DistributedSampler

Ensures each GPU gets different data batches:

```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Total number of GPUs
    rank=rank,                # This GPU's ID
    shuffle=True              # Shuffle data
)

dataloader = DataLoader(dataset, sampler=sampler)
```

### 4. Gradient Synchronization

DDP automatically synchronizes gradients using AllReduce:

```
GPU 0: [grad1, grad2, grad3]  ─┐
GPU 1: [grad4, grad5, grad6]  ─┤
GPU 2: [grad7, grad8, grad9]  ─┤─→ AllReduce
GPU 3: [grad10, grad11, grad12]─┘
         │
         ▼
All GPUs: [avg(grad1..10), avg(grad2..11), avg(grad3..12)]
```

## Usage

### Basic Example

```python
from distributed_training.data_parallel import DDPTrainer
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

# Create trainer
trainer = DDPTrainer(
    model=model,
    num_gpus=4,  # Use 4 GPUs
    mixed_precision=True  # Use AMP for faster training
)

# Train
trainer.train(train_loader, num_epochs=10)
```

### Launch Training

**Single Node, Multiple GPUs:**

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 train.py

# Or using python -m torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

**Multiple Nodes:**

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py

# Node 1
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py
```

## Advanced Features

### 1. Gradient Accumulation

Simulate larger batch sizes without requiring more memory:

```python
trainer = DDPTrainer(
    model=model,
    gradient_accumulation_steps=4  # Accumulate over 4 steps
)

# Effective batch size = batch_size * num_gpus * gradient_accumulation_steps
# Example: 32 * 4 * 4 = 512
```

### 2. Mixed Precision Training

Use FP16 for faster training and lower memory usage:

```python
trainer = DDPTrainer(
    model=model,
    mixed_precision=True  # Enable automatic mixed precision
)

# Benefits:
# - 2-3x faster training
# - 50% less GPU memory
# - Maintained accuracy with gradient scaling
```

### 3. Checkpointing

Automatic checkpoint saving and loading:

```python
# Training automatically saves checkpoints
trainer.train(train_loader, num_epochs=10, save_interval=2)

# Resume from checkpoint
epoch, loss = trainer.load_checkpoint("checkpoints/checkpoint_epoch_5.pt")
```

## Performance Tips

### 1. Batch Size Tuning

- **Rule of thumb**: Scale batch size linearly with number of GPUs
  - 1 GPU: batch_size = 32
  - 4 GPUs: batch_size = 32 (per GPU) = 128 total
  - 8 GPUs: batch_size = 32 (per GPU) = 256 total

### 2. Learning Rate Scaling

When increasing batch size, scale learning rate proportionally:

```python
base_lr = 0.001
num_gpus = 4
scaled_lr = base_lr * num_gpus  # Linear scaling rule
```

### 3. NCCL Optimization

Set environment variables for better performance:

```bash
export NCCL_DEBUG=INFO  # For debugging
export NCCL_IB_DISABLE=0  # Enable InfiniBand (if available)
export NCCL_SOCKET_IFNAME=eth0  # Specify network interface
```

### 4. Pin Memory

Enable pinned memory for faster CPU-to-GPU transfer:

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # Faster data transfer
    num_workers=4     # Parallel data loading
)
```

## Common Issues

### Issue 1: "Address already in use"

**Solution**: Kill existing processes or change port:

```bash
# Kill processes
pkill -9 python

# Or change port
export MASTER_PORT=29501
```

### Issue 2: Processes hang at initialization

**Solution**: Check network configuration:

```bash
# Test connectivity
ping <master_node_ip>

# Check firewall
sudo ufw allow 29500
```

### Issue 3: Out of memory

**Solutions**:
1. Reduce batch size
2. Enable mixed precision
3. Use gradient accumulation
4. Use gradient checkpointing

## Benchmarks

Performance scaling with data parallelism:

| GPUs | Batch Size | Throughput (samples/s) | Scaling Efficiency |
|------|------------|------------------------|-------------------|
| 1    | 32         | 100                    | -                 |
| 2    | 64         | 190                    | 95%               |
| 4    | 128        | 380                    | 95%               |
| 8    | 256        | 750                    | 94%               |

**Near-linear scaling!** Data parallelism is very efficient when:
- Model fits in single GPU memory
- Communication overhead is low
- Batch size scales with GPUs

## When to Use Data Parallelism

✅ **Use When:**
- Model fits in single GPU memory
- You want to train with larger batch sizes
- You have multiple GPUs available
- Simple implementation is preferred

❌ **Don't Use When:**
- Model is too large for single GPU (use Model Parallelism)
- Batch size is constrained by algorithm
- Communication overhead dominates (very small models)

## Further Reading

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed Training Best Practices](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
