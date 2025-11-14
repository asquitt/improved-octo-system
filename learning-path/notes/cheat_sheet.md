# Distributed Training Cheat Sheet

## Quick Commands

```bash
# Single GPU
python train.py

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 train.py

# Multi-node
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 train.py

# Check GPUs
nvidia-smi

# Monitor training
tensorboard --logdir=runs
```

## Code Snippets

### Initialize DDP
```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
```

### Wrap model in DDP
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().cuda()
model = DDP(model, device_ids=[rank])
```

### Distributed data loading
```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

### Save checkpoint (rank 0 only)
```python
if rank == 0:
    torch.save(model.state_dict(), 'checkpoint.pt')
```

### Synchronize processes
```python
dist.barrier()  # Wait for all processes
```

## Common Errors

### CUDA Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision

### Distributed hangs
- Check all processes reach sync points
- Verify network connectivity
- Increase timeout

### Slow training
- Check data loading (use num_workers>0)
- Profile with torch.profiler
- Ensure NCCL using GPUDirect
