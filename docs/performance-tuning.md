# Performance Tuning Guide

## Key Optimizations

### 1. Batch Size

Scale linearly with GPUs:
```python
base_batch_size = 32
num_gpus = 4
batch_size_per_gpu = base_batch_size  # Keep same per GPU
effective_batch_size = batch_size_per_gpu * num_gpus  # 128 total
```

### 2. Learning Rate

Linear scaling rule:
```python
base_lr = 0.001
scaled_lr = base_lr * num_gpus
```

### 3. Data Loading

```python
loader = create_distributed_dataloader(
    dataset,
    batch_size=32,
    num_workers=4,      # 4+ workers per GPU
    pin_memory=True,    # Faster CPU→GPU
    prefetch_factor=2,  # Prefetch batches
)
```

### 4. Mixed Precision

```python
trainer = DDPTrainer(
    model=model,
    mixed_precision=True  # 2-3x speedup
)
```

### 5. Gradient Accumulation

```python
trainer = DDPTrainer(
    model=model,
    gradient_accumulation_steps=4  # Larger effective batch
)
```

## Profiling

```python
from distributed_training.profiling import PerformanceTracker

tracker = PerformanceTracker()

with tracker.measure("forward"):
    output = model(data)

with tracker.measure("backward"):
    loss.backward()

tracker.print_stats()
```

## Common Bottlenecks

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| Data Loading | Low GPU util | Increase num_workers |
| Communication | Poor scaling | Larger batch size |
| Memory | OOM errors | Mixed precision, smaller batch |
| Compute | Slow training | Profile model, optimize ops |

## NCCL Optimization

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_SOCKET_IFNAME=eth0
```

## Monitoring

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Better monitoring
nvtop

# Profile training
python -m torch.utils.bottleneck train.py
```
