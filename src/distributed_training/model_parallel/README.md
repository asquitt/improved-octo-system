# Model Parallelism (Tensor Parallelism)

## Overview

Model Parallelism splits a **single model** across multiple GPUs. This is essential when:
- Model is too large to fit in single GPU memory
- You want to train models with billions of parameters
- Memory is the bottleneck, not compute

## How It Works

Unlike data parallelism (which replicates the model), model parallelism **partitions** the model:

```
┌─────────────────────────────────────────────────────┐
│              Single Large Model                      │
│         (Too big for 1 GPU!)                        │
└─────────────────────────────────────────────────────┘
                      │
                      │ Split Model
                      ▼
      ┌───────────────────────────────┐
      │                               │
┌─────▼─────┐                  ┌──────▼──────┐
│  GPU 0    │                  │   GPU 1     │
│ ┌───────┐ │                  │ ┌─────────┐ │
│ │Layer  │ │──────────────────▶│ │Layer    │ │
│ │1-25   │ │   Pass Activations│ │26-50    │ │
│ └───────┘ │                  │ └─────────┘ │
└───────────┘                  └─────────────┘
```

## Tensor Parallelism

Tensor parallelism splits **individual layers** across GPUs:

### Column Parallel Linear Layer

Splits the weight matrix **vertically** (along output dimension):

```
Normal Linear Layer: Y = X @ W + b

X: [batch, in_features]
W: [in_features, out_features]
Y: [batch, out_features]

Column Parallel:
┌──────────────────────────────────────┐
│         Weight Matrix W              │
│  [in_features, out_features]         │
└──────────────────────────────────────┘
        │                    │
        ▼                    ▼
    ┌───────┐           ┌────────┐
    │  W1   │  GPU 0    │   W2   │  GPU 1
    │[in,   │           │ [in,   │
    │ out/2]│           │  out/2]│
    └───┬───┘           └────┬───┘
        │                    │
        ▼                    ▼
     Y1 = X@W1            Y2 = X@W2
        │                    │
        └──────────┬─────────┘
                   ▼
          Y = [Y1 | Y2] (Concatenate)
```

**Key Properties:**
- Each GPU stores only part of weight matrix
- No communication during forward pass
- Output is concatenated across GPUs
- Memory per GPU: `1/N` of original (N = num GPUs)

### Row Parallel Linear Layer

Splits the weight matrix **horizontally** (along input dimension):

```
Row Parallel:
┌──────────────────────────────────────┐
│         Weight Matrix W              │
│  [in_features, out_features]         │
│                                      │
│  ┌────────────────────┐              │
│  │       W1           │  GPU 0       │
│  │  [in/2, out]       │              │
│  ├────────────────────┤              │
│  │       W2           │  GPU 1       │
│  │  [in/2, out]       │              │
│  └────────────────────┘              │
└──────────────────────────────────────┘

Input X is split: X = [X1 | X2]
GPU 0: Y1 = X1 @ W1
GPU 1: Y2 = X2 @ W2

Final: Y = Y1 + Y2 (AllReduce)
```

**Key Properties:**
- Input must be split across GPUs
- Each GPU computes partial result
- AllReduce sums results from all GPUs
- Memory per GPU: `1/N` of original

## Combining Column and Row Parallel

For maximum efficiency, alternate column and row parallel layers:

```
Input
  │
  ▼
┌─────────────────────────┐
│  Column Parallel        │  Splits output
│  (No communication)     │
└───────────┬─────────────┘
            │ (Split activations)
            ▼
┌─────────────────────────┐
│  ReLU (local)           │  Applied independently
└───────────┬─────────────┘
            │ (Still split)
            ▼
┌─────────────────────────┐
│  Row Parallel           │  Merges with AllReduce
│  (AllReduce needed)     │
└───────────┬─────────────┘
            │ (Full activations)
            ▼
          Output
```

**Benefits:**
- Only one AllReduce per layer pair
- Minimal communication overhead
- Maximum memory savings

## Usage Examples

### Example 1: Basic Tensor Parallel Layers

```python
from distributed_training.model_parallel import (
    ColumnParallelLinear,
    RowParallelLinear
)

# Column parallel layer (splits output)
layer1 = ColumnParallelLinear(
    in_features=1024,
    out_features=4096,  # Split across GPUs
    gather_output=False  # Keep output split
)

# Row parallel layer (merges input)
layer2 = RowParallelLinear(
    in_features=4096,  # Split input expected
    out_features=1024,
    input_is_parallel=True  # Input already split
)

# Forward pass
x = torch.randn(32, 1024)
h = layer1(x)  # Output split across GPUs
h = torch.relu(h)
y = layer2(h)  # Output gathered via AllReduce
```

### Example 2: Complete Model

```python
from distributed_training.model_parallel import TensorParallelModel

# Large model split across 4 GPUs
model = TensorParallelModel(
    input_size=1024,
    hidden_size=8192,  # 8GB per layer → 2GB per GPU
    output_size=1000,
    num_layers=4
)

# Check memory savings
from distributed_training.model_parallel.tensor_parallel import get_model_memory_usage

memory = get_model_memory_usage(model)
print(f"Model memory per GPU: {memory['total_size_gb']:.2f} GB")
# With 4 GPUs: ~4x memory reduction!
```

### Example 3: Launch Multi-GPU Training

```python
# train_model_parallel.py
import torch
import torch.distributed as dist
from distributed_training.model_parallel import TensorParallelModel

def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    # Create model (automatically sharded across GPUs)
    model = TensorParallelModel(
        input_size=1024,
        hidden_size=16384,  # Very large!
        output_size=1000
    ).cuda()

    # Train as normal
    optimizer = torch.optim.Adam(model.parameters())

    for data, target in dataloader:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    main()
```

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 train_model_parallel.py
```

## Communication Patterns

### Column Parallel
```
Forward:  No communication (each GPU computes independently)
Backward: AllReduce gradients for input
```

### Row Parallel
```
Forward:  AllReduce to sum partial outputs
Backward: No communication (gradients split naturally)
```

### Memory and Communication Trade-off

| Strategy | Memory per GPU | Communication | Best For |
|----------|---------------|---------------|----------|
| No Parallelism | 100% | None | Small models |
| Data Parallel | 100% | AllReduce gradients | Medium models |
| Model Parallel | 100%/N GPUs | Forward/Backward pass | Huge models |
| Hybrid (Both) | 100%/N GPUs | Both | Very large scale |

## Performance Considerations

### When Model Parallelism Helps

✅ **Good:**
- Model >> GPU memory (e.g., 100GB model, 40GB GPU)
- Large matrix multiplications (e.g., transformer FFN layers)
- High-bandwidth interconnect (NVLink, InfiniBand)

❌ **Bad:**
- Small models (communication overhead dominates)
- Limited interconnect bandwidth
- Many small operations (overhead per operation)

### Optimization Tips

1. **Minimize Communication:**
   ```python
   # Good: Alternate column and row parallel
   h = column_parallel(x)  # No comm
   h = activation(h)        # No comm
   y = row_parallel(h)      # One AllReduce

   # Bad: Gather after each layer
   h = column_parallel(x, gather_output=True)  # AllGather
   y = column_parallel(h, gather_output=True)  # AllGather again!
   ```

2. **Use NVLink/NVSwitch:**
   ```bash
   # Check NVLink connectivity
   nvidia-smi topo -m

   # GPUs should show NV# (NVLink) not PHB (PCIe)
   ```

3. **Fuse Operations:**
   ```python
   # Fuse activation with linear layer (no intermediate storage)
   class FusedLinear(nn.Module):
       def forward(self, x):
           x = self.linear(x)
           x = torch.nn.functional.gelu(x)  # Fused
           return x
   ```

## Memory Calculations

### Example: GPT-3 Style Model

```python
# Model config
n_layers = 96
hidden_size = 12288
n_heads = 96
vocab_size = 50257

# Memory per layer (roughly)
attention_weights = 4 * hidden_size * hidden_size  # Q, K, V, O projections
ffn_weights = 2 * hidden_size * (4 * hidden_size)  # Up and down projections
layer_memory = (attention_weights + ffn_weights) * 4  # 4 bytes per float32

total_memory = n_layers * layer_memory
print(f"Total model size: {total_memory / 1e9:.2f} GB")
# ~350GB!

# With 8 GPUs
per_gpu_memory = total_memory / 8
print(f"Per GPU with model parallel: {per_gpu_memory / 1e9:.2f} GB")
# ~44GB - fits in A100!
```

## Advanced: Hybrid Parallelism

Combine data and model parallelism for maximum scale:

```
┌──────────────────────────────────────────────┐
│            8 GPU Cluster                      │
│                                              │
│  Data Parallel Group 0    Data Parallel Group 1│
│  ┌──────┐  ┌──────┐      ┌──────┐  ┌──────┐│
│  │GPU 0 │  │GPU 1 │      │GPU 2 │  │GPU 3 ││
│  │Model │  │Model │      │Model │  │Model ││
│  │Part 1│  │Part 2│      │Part 1│  │Part 2││
│  └──────┘  └──────┘      └──────┘  └──────┘│
│      │         │              │         │   │
│      └─────────┼──────────────┼─────────┘   │
│          Model Parallel        Model Parallel│
│          (2 GPUs)              (2 GPUs)     │
│                                              │
│          Data Parallel (2 replicas)          │
└──────────────────────────────────────────────┘
```

## Comparison with Other Strategies

| Feature | Data Parallel | Model Parallel | Pipeline Parallel |
|---------|--------------|----------------|------------------|
| Model Replication | Full copies | Partitioned | Partitioned by layer |
| Memory per GPU | 100% | 100%/N | 100%/N stages |
| Communication | AllReduce gradients | Forward/backward pass | Sequential |
| Scalability | Limited by batch size | Limited by model structure | Limited by pipeline depth |
| Implementation | Simple | Medium | Complex |
| Efficiency | 95%+ | 70-90% | 80-95% |

## Debugging Tips

### Check Model Sharding

```python
# Verify each GPU has different parameters
for name, param in model.named_parameters():
    print(f"[Rank {dist.get_rank()}] {name}: {param.shape}")
```

### Monitor Communication

```bash
# Enable NCCL logging
export NCCL_DEBUG=INFO

# Run training
torchrun --nproc_per_node=4 train.py

# Look for AllReduce/AllGather operations
```

### Profile Memory

```python
import torch

# Before model creation
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# After model creation
model = TensorParallelModel(...)
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# After training step
loss.backward()
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Further Reading

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053) - Original tensor parallelism paper
- [PyTorch Tensor Parallel Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [Efficient Large-Scale Training](https://huggingface.co/docs/transformers/parallelism)
- [Model Parallelism Best Practices](https://github.com/NVIDIA/Megatron-LM)
