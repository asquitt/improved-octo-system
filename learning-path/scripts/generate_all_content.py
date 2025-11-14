#!/usr/bin/env python3
"""
Generate all learning path content

This script creates all the week READMEs, templates, exercises,
and reference materials for the complete 8-week course.

Usage:
    python scripts/generate_all_content.py
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent


def create_week_readme(week_num, title, topics, deliverable):
    """Create README for a specific week"""
    content = f"""# Week {week_num}: {title}

## 🎯 Learning Objectives

{topics}

## 📚 What You'll Build

**Deliverable:** {deliverable}

## 🚀 Getting Started

```bash
cd week-{week_num}
cat concepts.md      # Read theory
code starter_code.py  # Fill in code
python lab_*.py      # Run hands-on lab
python exercises.py  # Practice
```

## ✅ Completion Checklist

- [ ] Read concepts.md
- [ ] Complete starter_code.py
- [ ] Run lab successfully
- [ ] Finish exercises
- [ ] Can explain all concepts
- [ ] Ready for Week {week_num + 1}!

## 📊 Time Estimate

6-10 hours total

## 🆘 Need Help?

- Check `../notes/common_errors.md`
- Review `../solutions/week-{week_num}-solutions/`
- Re-read concepts if stuck

**Let's dive in! 🚀**
"""
    return content


def create_week_2():
    """Week 2: Data Parallel Training (DDP)"""
    topics = """
- Understand DistributedDataParallel (DDP)
- Initialize process groups correctly
- Implement multi-GPU training
- Handle distributed data loading
- Debug distributed training issues
    """
    deliverable = "Train ResNet on CIFAR-10 across 4 GPUs with linear speedup"

    readme = create_week_readme(2, "Data Parallel Training (DDP)", topics, deliverable)

    concepts = """# Week 2 Concepts: Distributed Data Parallel (DDP)

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
"""

    starter_code = """#!/usr/bin/env python3
\"\"\"
Week 2 Starter Code: Distributed Data Parallel (DDP)

TODO: Convert single-GPU code to multi-GPU with DDP
\"\"\"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    \"\"\"
    Initialize distributed training

    TODO: Set up process group
    \"\"\"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # TODO: Initialize process group
    dist._______(backend='______', rank=rank, world_size=world_size)

def cleanup():
    \"\"\"Clean up distributed training\"\"\"
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    \"\"\"
    Main DDP training function

    TODO: Implement DDP training loop
    \"\"\"
    # TODO: Setup
    setup(rank, world_size)

    # TODO: Set device for this rank
    device = torch.device(f'cuda:{______}')

    # TODO: Create model and wrap in DDP
    model = MyModel().to(device)
    model = ______(model, device_ids=[______])

    # TODO: Create DistributedSampler for data loading
    sampler = ______(train_dataset, num_replicas=world_size, rank=rank)

    # TODO: Training loop (similar to single-GPU)
    # ...

    cleanup()

def main():
    world_size = torch.cuda.device_count()

    # TODO: Spawn processes for each GPU
    mp.______(
        train_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
"""

    return {
        'README.md': readme,
        'concepts.md': concepts,
        'starter_code.py': starter_code,
    }


def create_week_3_to_8_summaries():
    """Create summaries for weeks 3-8"""
    weeks = {
        3: {
            'title': 'Model Parallel Training',
            'topics': 'Tensor parallelism, Pipeline parallelism, Hybrid strategies',
            'deliverable': 'Train large transformer with model parallelism'
        },
        4: {
            'title': 'Fully Sharded Data Parallel (FSDP)',
            'topics': 'FSDP architecture, Memory optimization, FSDP2, CPU offloading',
            'deliverable': 'Train GPT-2 scale model with FSDP'
        },
        5: {
            'title': 'Advanced Optimizations',
            'topics': 'Mixed precision, torch.compile, Gradient compression, Activation checkpointing',
            'deliverable': '3x faster training with optimization stack'
        },
        6: {
            'title': 'Production Best Practices',
            'topics': 'Checkpointing, Fault tolerance, Monitoring, Distributed data loading',
            'deliverable': 'Production-ready training pipeline'
        },
        7: {
            'title': 'Deployment & Scaling',
            'topics': 'Multi-node training, Cloud deployment, Slurm, Cost optimization',
            'deliverable': 'Multi-node training job on cloud'
        },
        8: {
            'title': 'Advanced Topics & Capstone',
            'topics': 'ZeRO, Custom kernels, Profiling, Debugging',
            'deliverable': 'Complete capstone - Train your own LLM'
        },
    }

    content = {}
    for week, info in weeks.items():
        content[week] = create_week_readme(
            week,
            info['title'],
            info['topics'],
            info['deliverable']
        )
    return content


def create_templates():
    """Create code templates"""
    templates = {}

    # Training loop template
    templates['training_loop.py'] = """#!/usr/bin/env python3
\"\"\"Template for complete training loop\"\"\"

def train_model(model, train_loader, val_loader, config):
    \"\"\"
    Complete training pipeline template

    Fill in your custom logic while keeping structure
    \"\"\"
    # Setup
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    criterion = create_criterion(config)

    # Training loop
    for epoch in range(config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion
        )

        # Step scheduler
        scheduler.step()

        # Checkpoint
        if val_metrics['loss'] < best_loss:
            save_checkpoint(model, optimizer, epoch)

        # Log
        log_metrics(train_metrics, val_metrics, epoch)

    return model
"""

    # Model template
    templates['model_template.py'] = """#!/usr/bin/env python3
\"\"\"Template for model definition\"\"\"

import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers
        pass

    def forward(self, x):
        # Forward pass
        return x

    def compute_loss(self, outputs, targets):
        # Custom loss computation
        pass
"""

    # Config template
    templates['config_template.py'] = """#!/usr/bin/env python3
\"\"\"Template for configuration\"\"\"

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model
    model_name: str = 'resnet50'
    hidden_size: int = 768
    num_layers: int = 12

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4

    # Distributed
    world_size: int = 4
    backend: str = 'nccl'

    # Optimization
    use_amp: bool = True
    use_compile: bool = True
    gradient_clip: float = 1.0
"""

    return templates


def create_notes():
    """Create reference notes"""
    notes = {}

    notes['cheat_sheet.md'] = """# Distributed Training Cheat Sheet

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
"""

    notes['glossary.md'] = """# Distributed Training Glossary

## Terms

**Rank:** Unique ID for each process (0, 1, 2, ...)

**World Size:** Total number of processes

**Local Rank:** Rank within a node

**Process Group:** Collection of communicating processes

**Backend:** Communication library (NCCL, Gloo, MPI)

**AllReduce:** Combine values from all processes

**Broadcast:** Send from one to all

**Scatter/Gather:** Distribute/collect data

**DDP:** DistributedDataParallel

**FSDP:** Fully Sharded Data Parallel

**Pipeline Parallel:** Split model layers across devices

**Tensor Parallel:** Split tensors across devices

**Gradient Accumulation:** Sum gradients over multiple steps

**Mixed Precision:** Use FP16/BF16 for speed, FP32 for stability

**Checkpoint:** Saved model state for recovery

**Epoch:** One pass through entire dataset

**Batch:** Subset of data processed together

**Learning Rate:** Step size for optimizer

**Scheduler:** Adjusts learning rate during training
"""

    notes['best_practices.md'] = """# Distributed Training Best Practices

## 1. Start Simple
- Get single-GPU working first
- Add distributed features incrementally
- Test at each step

## 2. Data Loading
- Use num_workers >= 4
- Pin memory for faster CPU-GPU transfer
- Use DistributedSampler for DDP
- Prefetch data

## 3. Checkpointing
- Save from rank 0 only
- Include optimizer state
- Save every N epochs
- Keep best and latest

## 4. Debugging
- Print from rank 0 only
- Set deterministic seeds
- Use NCCL_DEBUG=INFO
- Start with small model

## 5. Performance
- Use mixed precision
- Enable torch.compile
- Profile bottlenecks
- Overlap communication

## 6. Monitoring
- Track loss, accuracy, throughput
- Monitor GPU utilization
- Log learning rate
- Use TensorBoard/wandb

## 7. Reproducibility
- Set all random seeds
- Use deterministic algorithms
- Save full configuration
- Version control code
"""

    notes['common_errors.md'] = """# Common Errors and Solutions

## CUDA Errors

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision
- Clear cache: torch.cuda.empty_cache()

### Device-side assert
```
RuntimeError: CUDA error: device-side assert triggered
```
**Solutions:**
- Run on CPU to get better error message
- Check label ranges match model output
- Verify data types

## Distributed Errors

### Timeout
```
RuntimeError: ProcessGroupNCCL timeout
```
**Solutions:**
- Increase timeout in init_process_group
- Check network connectivity
- Verify all ranks reach synchronization points

### Rank mismatch
```
RuntimeError: rank not in the process group
```
**Solutions:**
- Ensure consistent world_size
- Check environment variables
- Verify process group initialization

### Hangs at barrier
**Solutions:**
- Check all ranks execute same code
- Remove conditional barriers
- Verify data loading doesn't hang

## Data Errors

### Shape mismatch
```
RuntimeError: size mismatch
```
**Solutions:**
- Print shapes at each step
- Verify transforms
- Check batch dimensions

### DataLoader hang
**Solutions:**
- Reduce num_workers
- Check data path exists
- Simplify transforms first

## Installation Errors

### Import errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solutions:**
- pip install torch torchvision
- Check virtual environment
- Verify Python version

### CUDA not available
**Solutions:**
- Install CUDA-enabled PyTorch
- Check GPU drivers: nvidia-smi
- Verify CUDA version matches PyTorch
"""

    return notes


def main():
    """Generate all content"""
    print("🚀 Generating learning path content...")

    # Create week 2
    print("\n📝 Creating Week 2...")
    week2_dir = BASE_DIR / 'week-2'
    week2_dir.mkdir(exist_ok=True)

    week2_content = create_week_2()
    for filename, content in week2_content.items():
        filepath = week2_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ {filename}")

    # Create weeks 3-8 summaries
    print("\n📝 Creating Weeks 3-8...")
    week_summaries = create_week_3_to_8_summaries()
    for week, content in week_summaries.items():
        week_dir = BASE_DIR / f'week-{week}'
        week_dir.mkdir(exist_ok=True)

        filepath = week_dir / 'README.md'
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Week {week} README")

    # Create templates
    print("\n📝 Creating templates...")
    templates_dir = BASE_DIR / 'templates'
    templates_dir.mkdir(exist_ok=True)

    templates = create_templates()
    for filename, content in templates.items():
        filepath = templates_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ {filename}")

    # Create notes
    print("\n📝 Creating reference notes...")
    notes_dir = BASE_DIR / 'notes'
    notes_dir.mkdir(exist_ok=True)

    notes = create_notes()
    for filename, content in notes.items():
        filepath = notes_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ {filename}")

    print("\n✅ All content generated successfully!")
    print("\n📚 Next steps:")
    print("  1. cd learning-path/week-1")
    print("  2. cat README.md")
    print("  3. python starter_code.py")
    print("\n🎉 Happy learning!")


if __name__ == '__main__':
    main()
