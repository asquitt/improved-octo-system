# Quick Start Guide

Get started with distributed training in 5 minutes!

## 1. Install

```bash
pip install -e .
```

## 2. Run Your First Example

```bash
# Data parallel training (recommended start)
torchrun --nproc_per_node=2 examples/01_data_parallel_simple.py
```

## 3. Understand What Happened

The example:
- Created a simple CNN model
- Replicated it on 2 GPUs
- Trained with different data batches on each GPU
- Synchronized gradients automatically

## 4. Try Your Own Model

```python
import torch.nn as nn
from distributed_training.data_parallel import DDPTrainer

# Your model
model = YourModel()

# Create trainer
trainer = DDPTrainer(model=model, num_gpus=2)

# Train
trainer.train(your_dataloader, num_epochs=10)
```

Launch with:
```bash
torchrun --nproc_per_node=2 your_script.py
```

## 5. Next Steps

- **Learn more**: Read [docs/getting-started.md](docs/getting-started.md)
- **Examples**: Explore [examples/](examples/)
- **Advanced**: Try model parallelism for huge models

## Common Issues

**No GPUs detected?**
```bash
nvidia-smi  # Check if GPUs are visible
```

**Import errors?**
```bash
pip install -e .  # Install in development mode
```

**Port already in use?**
```bash
export MASTER_PORT=29501
```

That's it! Happy training! 🚀
