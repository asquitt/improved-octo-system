# Getting Started

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd improved-octo-system
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 3. Verify Installation

```bash
python -c "import distributed_training; print(distributed_training.__version__)"
```

## Quick Start

### Your First Distributed Training

```python
import torch
import torch.nn as nn
from distributed_training.data_parallel import DDPTrainer

# 1. Define model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 2. Create trainer
trainer = DDPTrainer(
    model=model,
    num_gpus=4
)

# 3. Train
trainer.train(train_loader, num_epochs=10)
```

### Launch Training

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py
```

## Next Steps

- Read [Data Parallelism Guide](data-parallelism.md)
- Try [Examples](../examples/)
- Check [Performance Tuning](performance-tuning.md)
