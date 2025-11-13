# Pipeline Parallelism

## Overview

Pipeline parallelism partitions the model **by layers** and processes multiple micro-batches simultaneously.

```
Model: Layer1 → Layer2 → Layer3 → Layer4

Pipeline Split:
GPU 0: Layer1
GPU 1: Layer2
GPU 2: Layer3
GPU 3: Layer4

Execution Timeline:
Time → F1  F2  F3  F4  B1  B2  B3  B4
GPU0: [F1][F2][F3][F4][B1][B2][B3][B4]
GPU1:     [F1][F2][F3][F4][B1][B2][B3][B4]
GPU2:         [F1][F2][F3][F4][B1][B2][B3][B4]
GPU3:             [F1][F2][F3][F4][B1][B2][B3][B4]
```

## Key Concepts

- **Micro-batches**: Split batch into smaller chunks
- **Bubble Time**: Idle time at pipeline start/end
- **Efficiency**: More micro-batches = less bubble time

## Usage

```python
from distributed_training.pipeline_parallel import PipelineTrainer

model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
)

trainer = PipelineTrainer(
    model=model,
    num_stages=4,
    micro_batch_size=8
)

trainer.train(train_loader, num_epochs=10)
```

Launch: `torchrun --nproc_per_node=4 train.py`

## Efficiency Formula

`Efficiency = (M / (M + S - 1)) * 100%`

Where M = micro-batches, S = stages

| Stages | Micro-batches | Efficiency |
|--------|--------------|------------|
| 4      | 4            | 57%        |
| 4      | 8            | 73%        |
| 4      | 16           | 84%        |

**Tip**: Use at least 4× micro-batches as stages!
