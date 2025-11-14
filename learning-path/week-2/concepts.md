# Week 2 Concepts: Distributed Data Parallel (DDP)

## What is DDP?

Distributed Data Parallel replicates your model across multiple GPUs and synchronizes
gradients after each backward pass.

### Key Concepts:

1. **Process Groups:** Each GPU runs in its own process
2. **All-Reduce:** Gradients are averaged across all GPUs
3. **Synchronized Updates:** All models stay identical

### Basic DDP Pattern:

```python
# Initialize distributed training
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
device = torch.device(f'cuda:{rank}')

# Wrap model in DDP
model = MyModel().to(device)
model = DDP(model, device_ids=[rank])

# Training loop (same as single-GPU!)
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()  # Gradients automatically synchronized!
    optimizer.step()
```

## Launching Multi-GPU Training

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py

# Multi-node
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 train_ddp.py
```

## Common Pitfalls

1. **Forgetting to set rank:** Each process needs unique CUDA device
2. **Non-deterministic seeds:** Set seed per rank for reproducibility
3. **Global batch size:** Effective batch = batch_per_gpu * num_gpus
4. **Printing from all ranks:** Only print from rank 0
5. **Data shuffling:** Use DistributedSampler

## Performance Tips

- Use NCCL backend for NVIDIA GPUs
- Pin memory in DataLoader
- Enable gradient_as_bucket_view=True
- Find optimal bucket_cap_mb
