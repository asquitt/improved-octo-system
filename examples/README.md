# Examples

This directory contains practical examples demonstrating various distributed training strategies.

## Quick Start

Each example can be run with `torchrun`. Make sure you have installed the package first:

```bash
pip install -e .
```

## Examples

### 1. Data Parallel Training (`01_data_parallel_simple.py`)

**What it teaches**: Basic multi-GPU training with PyTorch DDP

**Best for**: Models that fit in single GPU memory

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 examples/01_data_parallel_simple.py

# Single node, 2 GPUs
torchrun --nproc_per_node=2 examples/01_data_parallel_simple.py
```

**Key Features**:
- DistributedDataParallel (DDP) setup
- Automatic gradient synchronization
- Mixed precision training
- Gradient accumulation

---

### 2. Model Parallel Training (`02_model_parallel_large_model.py`)

**What it teaches**: Splitting large models across GPUs

**Best for**: Models too large for single GPU memory

```bash
torchrun --nproc_per_node=4 examples/02_model_parallel_large_model.py
```

**Key Features**:
- Tensor parallelism
- Column and row parallel layers
- Memory-efficient for huge models

---

### 3. Pipeline Parallel Training (`03_pipeline_parallel_transformer.py`)

**What it teaches**: Layer-wise model partitioning with micro-batch pipelining

**Best for**: Very deep models (transformers, ResNets)

```bash
torchrun --nproc_per_node=4 examples/03_pipeline_parallel_transformer.py
```

**Key Features**:
- Pipeline stage partitioning
- Micro-batch scheduling
- Efficiency calculation

---

### 4. Complete Training Demo (`04_complete_training_demo.py`)

**What it teaches**: All features together

**Best for**: Production training setup

```bash
torchrun --nproc_per_node=2 examples/04_complete_training_demo.py
```

**Key Features**:
- Checkpointing
- Performance profiling
- Fault tolerance
- Efficient data loading

---

## Multi-Node Training

For training across multiple machines:

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    examples/01_data_parallel_simple.py

# Node 1
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    examples/01_data_parallel_simple.py
```

## Tips

1. **Start Simple**: Begin with example 1 (data parallel)
2. **Check GPUs**: Verify GPUs are visible: `nvidia-smi`
3. **Monitor**: Use `nvtop` or `nvidia-smi` to monitor GPU usage
4. **Adjust Batch Size**: Increase batch size proportionally with GPUs
5. **Scale Learning Rate**: Use linear scaling rule: `lr = base_lr * num_gpus`

## Troubleshooting

**"Address already in use"**:
```bash
pkill -9 python
# or change port
export MASTER_PORT=29501
```

**Out of memory**:
- Reduce batch size
- Enable mixed precision
- Use gradient accumulation

**No speedup**:
- Check data loading (use `num_workers > 0`)
- Verify model is large enough to benefit
- Check for bottlenecks with profiler

## Further Customization

All examples are self-contained and easy to modify:
- Change model architecture
- Swap datasets
- Adjust hyperparameters
- Add custom metrics

Happy training! 🚀
