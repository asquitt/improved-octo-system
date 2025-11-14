# Week 1 Concepts: Foundation of Distributed Training

**Read Time:** 30-45 minutes

---

## 🎯 Why Distributed Training?

### The Problem

Imagine you want to train GPT-4 scale models:
- **175 billion parameters** = ~350 GB in FP16
- **Single A100 GPU** = 80GB memory ❌ **Won't fit!**
- **Training time:** Years on single GPU ❌ **Too slow!**

**Solution:** Distribute training across many GPUs/nodes.

### Real-World Examples

**OpenAI GPT-3:**
- 175B parameters
- Trained on 10,000+ GPUs
- 3.14×10²³ FLOPs
- Cost: ~$4.6M in compute

**Meta LLaMA-2:**
- 70B parameters
- Trained on 2,000+ GPUs
- Finished in weeks instead of years

**Your Project:**
- Maybe 1B parameters
- Train on 4-8 GPUs
- Finish in days instead of months

---

## 📊 When Do You Need Distributed Training?

### Decision Tree

```
Is your model < 8GB?
├─ Yes → Is training time < 1 day?
│  ├─ Yes → ✅ Single GPU is fine
│  └─ No → ⚠️ Consider data parallel
└─ No → Does model fit on 1 GPU?
   ├─ Yes → ⚠️ Use data parallel for speed
   └─ No → ❌ Must use model parallel/FSDP
```

### Size Thresholds

| Model Size | Strategy | GPUs Needed |
|-----------|----------|-------------|
| < 1GB | Single GPU | 1 |
| 1-8GB | Single GPU or DDP | 1-4 |
| 8-24GB | DDP recommended | 2-8 |
| 24-100GB | FSDP or Model Parallel | 4-16 |
| 100GB+ | FSDP + Pipeline | 16-100+ |

---

## 🧩 Types of Parallelism

### 1. Data Parallel (Most Common)

**Idea:** Same model on each GPU, different data batches

```
GPU 0: Model Copy 1 → Batch 0 (images 0-31)
GPU 1: Model Copy 2 → Batch 1 (images 32-63)
GPU 2: Model Copy 3 → Batch 2 (images 64-95)
GPU 3: Model Copy 4 → Batch 3 (images 96-127)

After forward/backward:
→ Average all gradients
→ Update all model copies
→ Repeat
```

**Pros:**
- ✅ Easy to implement
- ✅ Works for most models
- ✅ Scales well to 8-16 GPUs
- ✅ No code changes to model

**Cons:**
- ❌ Model must fit on single GPU
- ❌ Communication overhead grows
- ❌ Batch size limited by global memory

**Use When:**
- Model fits on 1 GPU
- Want to speed up training
- Have 2-16 GPUs available

---

### 2. Model Parallel

**Idea:** Different parts of model on different GPUs

#### 2a. Tensor Parallelism (Wide Models)

```
GPU 0: Columns 0-255 of Linear layer
GPU 1: Columns 256-511 of Linear layer
GPU 2: Columns 512-767 of Linear layer
GPU 3: Columns 768-1023 of Linear layer

Forward: Split computation, then gather
Backward: Gather gradients, then split
```

**Example:** Splitting a Transformer attention head

```python
# Single GPU
attention = MultiHeadAttention(embed_dim=1024, num_heads=16)

# Tensor Parallel (4 GPUs)
# GPU 0: heads 0-3
# GPU 1: heads 4-7
# GPU 2: heads 8-11
# GPU 3: heads 12-15
```

**Pros:**
- ✅ Trains models too large for 1 GPU
- ✅ High GPU utilization
- ✅ Works for wide models

**Cons:**
- ❌ Requires model code changes
- ❌ Communication every layer
- ❌ Complex to implement

---

#### 2b. Pipeline Parallelism (Deep Models)

**Idea:** Sequential layers on different GPUs

```
GPU 0: Layers 1-6   (Input → Transformer blocks 1-6)
GPU 1: Layers 7-12  (Transformer blocks 7-12)
GPU 2: Layers 13-18 (Transformer blocks 13-18)
GPU 3: Layers 19-24 (Transformer blocks 19-24 → Output)

Data flows: GPU 0 → GPU 1 → GPU 2 → GPU 3
```

**Micro-batching:** Split each batch into smaller chunks

```
Batch of 64:
- Micro-batch 1 (16 samples): GPU 0 → GPU 1 → GPU 2 → GPU 3
- Micro-batch 2 (16 samples): GPU 0 → GPU 1 → GPU 2 → GPU 3
- Micro-batch 3 (16 samples): GPU 0 → GPU 1 → GPU 2 → GPU 3
- Micro-batch 4 (16 samples): GPU 0 → GPU 1 → GPU 2 → GPU 3
```

**Pros:**
- ✅ Trains very deep models
- ✅ Less communication than tensor parallel
- ✅ Natural for sequential models

**Cons:**
- ❌ GPU bubble time (idle periods)
- ❌ Requires careful scheduling
- ❌ More complex than data parallel

---

### 3. Fully Sharded Data Parallel (FSDP)

**Idea:** Shard model parameters, gradients, and optimizer states across GPUs

```
Model has 1B parameters:

GPU 0: Parameters[0:250M] + Full forward/backward
GPU 1: Parameters[250M:500M] + Full forward/backward
GPU 2: Parameters[500M:750M] + Full forward/backward
GPU 3: Parameters[750M:1B] + Full forward/backward

During forward:
- Each GPU gathers needed parameters from others
- Computes its portion
- Discards unneeded parameters

Memory savings: 4x (with 4 GPUs)
```

**Pros:**
- ✅ Massive memory savings
- ✅ Can train huge models
- ✅ Relatively simple to use
- ✅ Scales to 100+ GPUs

**Cons:**
- ❌ More communication than DDP
- ❌ Requires PyTorch 1.11+
- ❌ Some overhead for small models

---

## 📡 Communication Patterns

### 1. AllReduce (Gradient Averaging)

**What:** Combine values from all GPUs, send result to all

```
Before:
GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]
GPU 2: [7, 8, 9]

After AllReduce (sum):
GPU 0: [12, 15, 18]
GPU 1: [12, 15, 18]
GPU 2: [12, 15, 18]

After AllReduce (mean):
GPU 0: [4, 5, 6]
GPU 1: [4, 5, 6]
GPU 2: [4, 5, 6]
```

**Used for:** Averaging gradients in data parallel training

### 2. Broadcast

**What:** Send data from one GPU to all others

```
Before:
GPU 0: [1, 2, 3]  ← Source
GPU 1: [?, ?, ?]
GPU 2: [?, ?, ?]

After Broadcast:
GPU 0: [1, 2, 3]
GPU 1: [1, 2, 3]
GPU 2: [1, 2, 3]
```

**Used for:** Distributing model weights, hyperparameters

### 3. Scatter/Gather

**Scatter:** Split data from one GPU to all
**Gather:** Collect data from all GPUs to one

```
Scatter:
GPU 0: [1,2,3,4,5,6,7,8,9] → [1,2,3]
GPU 1: [?, ?, ?, ?, ?, ?] → [4,5,6]
GPU 2: [?, ?, ?, ?, ?, ?] → [7,8,9]

Gather:
GPU 0: [1,2,3] → [1,2,3,4,5,6,7,8,9]
GPU 1: [4,5,6] →
GPU 2: [7,8,9] →
```

**Used for:** Data distribution, result collection

---

## 🔧 Hardware Considerations

### GPU Memory Hierarchy

```
Fastest ↑
-------
Registers: ~20 MB, <1 cycle
L1 Cache: ~128 KB per SM, ~1 cycle
L2 Cache: ~40 MB, ~200 cycles
HBM Memory: ~80 GB, ~300 cycles
-------
Slowest ↓

Between GPUs:
NVLink: ~600 GB/s (A100)
PCIe 4.0: ~64 GB/s
Network: ~100 Gbps (InfiniBand)
```

### Communication Overhead

**Rule of Thumb:**
- Intra-node (same machine): ~100 GB/s
- Inter-node (network): ~10-100 Gbps
- **Ratio:** 10-100x slower across nodes

**Implications:**
- Minimize all-to-all communication
- Overlap communication with computation
- Use gradient compression for multi-node

---

## 📈 Scaling Efficiency

### Linear Scaling (Ideal)

```
1 GPU:  100 samples/sec
2 GPUs: 200 samples/sec  ✅ 100% efficient
4 GPUs: 400 samples/sec  ✅ 100% efficient
8 GPUs: 800 samples/sec  ✅ 100% efficient
```

### Reality

```
1 GPU:  100 samples/sec
2 GPUs: 190 samples/sec  (95% efficient)
4 GPUs: 360 samples/sec  (90% efficient)
8 GPUs: 640 samples/sec  (80% efficient)
16 GPUs: 1120 samples/sec (70% efficient)
```

**Why not perfect?**
- Communication overhead
- Synchronization points
- Load imbalance
- Startup costs

### Strong vs Weak Scaling

**Strong Scaling:** Fixed total batch size
```
1 GPU: 128 batch → 100 samples/sec
2 GPUs: 128 batch (64 each) → 180 samples/sec
4 GPUs: 128 batch (32 each) → 320 samples/sec
```
*Problem:* Smaller per-GPU batches = less efficient

**Weak Scaling:** Fixed per-GPU batch size
```
1 GPU: 64 batch → 100 samples/sec
2 GPUs: 128 batch (64 each) → 195 samples/sec
4 GPUs: 256 batch (64 each) → 380 samples/sec
```
*Better efficiency, but changes training dynamics*

---

## 🎓 Key Terminology

### Process
A running instance of Python (one per GPU usually)

### Rank
Unique ID for each process (0, 1, 2, 3 for 4 processes)

### World Size
Total number of processes

### Local Rank
Rank within a node (for multi-node setups)

### Process Group
Collection of processes that communicate

### Backend
Communication library (NCCL for NVIDIA GPUs)

### Example:
```python
# 2 nodes, 4 GPUs each = 8 total processes

Node 0:
  GPU 0: rank=0, local_rank=0
  GPU 1: rank=1, local_rank=1
  GPU 2: rank=2, local_rank=2
  GPU 3: rank=3, local_rank=3

Node 1:
  GPU 4: rank=4, local_rank=0
  GPU 5: rank=5, local_rank=1
  GPU 6: rank=6, local_rank=2
  GPU 7: rank=7, local_rank=3

world_size = 8
```

---

## 🚀 Single GPU Baseline (This Week)

Before distributed training, master single-GPU training:

### Complete Training Loop

```python
# 1. Setup
device = torch.device('cuda')
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 2. Training Loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Compute metrics

# 3. Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')
```

---

## 💡 Key Takeaways

1. **Distributed training is necessary** for large models and datasets
2. **Data parallel is most common** and easiest to implement
3. **Model parallel needed** when model doesn't fit on 1 GPU
4. **FSDP combines benefits** of data and model parallel
5. **Communication is the bottleneck** in most distributed training
6. **Master single-GPU first** before going distributed
7. **Measure everything** - know your bottlenecks

---

## 🎯 Next Steps

Now that you understand the concepts:

1. ✅ Complete `starter_code.py` - Build a single-GPU trainer
2. ✅ Run `lab_single_gpu.py` - Hands-on experience
3. ✅ Do `exercises.py` - Practice makes perfect
4. ✅ Review `checkpoints.md` - Verify understanding

**Then:** Move to Week 2 and implement your first DDP training!

---

## 📚 Further Reading

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Horovod Paper](https://arxiv.org/abs/1802.05799) - Uber's distributed training
- [ZeRO Paper](https://arxiv.org/abs/1910.02054) - Microsoft's optimizer
- [Megatron-LM](https://arxiv.org/abs/1909.08053) - NVIDIA's model parallel

**Ready to code?** Open `starter_code.py` and let's build! 🚀
