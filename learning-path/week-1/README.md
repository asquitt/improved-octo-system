# Week 1: Foundation - Understanding Distributed Training

**Welcome to Week 1!** 🎉

This week lays the foundation for everything you'll learn. You'll understand **why** distributed training exists, **when** to use it, and build your first baseline single-GPU training script.

---

## 🎯 Learning Objectives

By the end of this week, you will:

- ✅ Understand what distributed training is and why it's needed
- ✅ Know the difference between data, model, and pipeline parallelism
- ✅ Build a complete single-GPU training loop from scratch
- ✅ Implement training monitoring and metrics
- ✅ Recognize when to use distributed training
- ✅ Set up your development environment properly

---

## 📚 Topics Covered

### 1. Why Distributed Training?
- Model size exceeds single GPU memory
- Training time is too long
- Larger batch sizes improve performance
- Research and production needs

### 2. Types of Parallelism
- **Data Parallel:** Same model, different data across GPUs
- **Model Parallel:** Different parts of model on different GPUs
- **Pipeline Parallel:** Sequential layers across GPUs
- **Hybrid:** Combinations of above

### 3. Communication Patterns
- AllReduce (gradient averaging)
- Broadcast (parameter sharing)
- Scatter/Gather (data distribution)
- Point-to-point communication

### 4. Single-GPU Baseline
- Training loop structure
- Forward and backward passes
- Optimization and metrics
- Checkpointing basics

---

## 📂 Files in This Week

```
week-1/
├── README.md              # This file - week overview
├── concepts.md            # Detailed theory and explanations
├── starter_code.py        # YOUR MAIN TASK - fill in the blanks
├── lab_single_gpu.py      # Hands-on lab exercise
├── exercises.py           # Practice problems
├── checkpoints.md         # Verify your understanding
├── notes.md               # Key takeaways and tips
└── data/                  # Sample datasets (downloaded on first run)
```

---

## 🚀 Getting Started

### Step 1: Read the Concepts (30 mins)

```bash
cat concepts.md
```

This explains the theory behind distributed training.

### Step 2: Complete Starter Code (2-3 hours)

```bash
# Open in your editor
code starter_code.py
```

**Your Task:** Fill in all the `TODO` sections and `______` blanks.

### Step 3: Run Your Code

```bash
python starter_code.py
```

**Expected Output:**
```
Training MNIST Classifier
Epoch 1/5: 100%|████████| 938/938 [00:45<00:00, 20.8it/s, loss=0.234]
  Train Loss: 0.234, Train Acc: 92.3%
  Val Loss: 0.145, Val Acc: 95.6%
Epoch 2/5: ...
```

### Step 4: Complete the Lab (2 hours)

```bash
python lab_single_gpu.py
```

This gives you hands-on experience with a complete training pipeline.

### Step 5: Practice with Exercises (1-2 hours)

```bash
python exercises.py
```

### Step 6: Verify Learning

```bash
cat checkpoints.md
```

Answer the questions to verify you understand the concepts.

---

## 📝 What You'll Build

### Main Task: MNIST Classifier

You'll build a complete training pipeline for MNIST digit recognition:

**Components:**
- Model definition (CNN)
- Data loading and preprocessing
- Training loop with progress bars
- Validation and metrics
- Model checkpointing
- Learning rate scheduling
- Early stopping

**Performance Target:**
- >95% validation accuracy
- <0.15 validation loss
- Completes in <10 minutes on single GPU

---

## 🎯 Success Criteria

You've mastered Week 1 when you can:

- [ ] Explain why distributed training is needed
- [ ] Describe data vs model vs pipeline parallelism
- [ ] Write a complete training loop from scratch
- [ ] Implement proper validation and metrics
- [ ] Save and load checkpoints correctly
- [ ] Monitor training with progress bars
- [ ] Recognize when single-GPU isn't enough

---

## 💡 Key Concepts to Master

### 1. Training Loop Structure

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch)
            # Compute metrics
```

### 2. When to Use Distributed Training

**Use distributed training when:**
- ✅ Model doesn't fit on single GPU (>24GB)
- ✅ Training takes >1 day on single GPU
- ✅ Dataset is very large (>100GB)
- ✅ Need faster iteration cycles
- ✅ Want to explore larger batch sizes

**Stick with single GPU when:**
- ❌ Model fits comfortably (<8GB)
- ❌ Training finishes in hours
- ❌ Debugging and development
- ❌ Small datasets

---

## 📊 Time Estimates

| Activity | Time | Difficulty |
|----------|------|-----------|
| Read concepts.md | 30 mins | ⭐ Easy |
| Complete starter_code.py | 2-3 hours | ⭐⭐ Medium |
| Run lab_single_gpu.py | 2 hours | ⭐⭐ Medium |
| Complete exercises.py | 1-2 hours | ⭐⭐ Medium |
| Review checkpoints.md | 30 mins | ⭐ Easy |

**Total:** 6-8 hours

---

## 🆘 Common Issues

### Issue 1: CUDA Out of Memory
```python
# Solution: Reduce batch size
batch_size = 64  # Try 32, 16, or 8
```

### Issue 2: Training Too Slow
```python
# Solution: Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Issue 3: Loss Not Decreasing
```python
# Solutions:
# 1. Check learning rate (try 0.001, 0.0001)
# 2. Verify labels are correct
# 3. Check data normalization
# 4. Use simpler model first
```

### Issue 4: Validation Worse Than Training
```python
# Solution: Add regularization
model.add_module('dropout', nn.Dropout(0.5))

# Or reduce model complexity
```

---

## 📚 Additional Resources

### This Week's Reading

1. **concepts.md** - Must read! Core theory
2. **notes.md** - Quick reference and tips
3. **Main docs:** `../../docs/QUICKSTART.md`

### External Resources

1. [PyTorch Tutorial: Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
2. [Understanding Training Loops](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
3. [Effective PyTorch](https://github.com/vahidk/EffectivePyTorch)

---

## ✅ Completion Checklist

- [ ] Read concepts.md thoroughly
- [ ] Completed starter_code.py (all TODOs done)
- [ ] Code runs without errors
- [ ] Achieved >95% validation accuracy
- [ ] Completed lab_single_gpu.py
- [ ] Finished all exercises
- [ ] Can explain all checkpoints
- [ ] Reviewed notes.md
- [ ] Ready for Week 2!

---

## 🎓 Next Week Preview

**Week 2: Data Parallel Training (DDP)**

Next week, you'll take your single-GPU code and scale it to multiple GPUs using Distributed Data Parallel (DDP). You'll learn:
- Process groups and initialization
- Gradient synchronization
- Multi-GPU data loading
- Debugging distributed code

**Get excited!** 🚀

---

## 💬 Questions to Ponder

As you work through this week, think about:

1. What are the bottlenecks in single-GPU training?
2. How would you profile where time is spent?
3. When does the dataset become too large for memory?
4. How do batch size and learning rate relate?
5. What metrics matter most for your use case?

---

## 🎯 Pro Tips

1. **Start Simple:** Get MNIST working before moving to complex datasets
2. **Use Debugger:** Don't just print, use pdb or IDE debugger
3. **Save Often:** Checkpoint every epoch during development
4. **Monitor Everything:** Track loss, accuracy, learning rate, time
5. **Version Control:** Commit working code frequently

---

**Ready?** Let's dive into `concepts.md` and start building!

```bash
cat concepts.md
code starter_code.py
```

**Happy coding! 🎉**
