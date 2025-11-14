# Getting Started with the Learning Path

**Welcome!** This guide will help you set up your environment and start your distributed training learning journey.

---

## 🎯 What You'll Learn

This 8-week course teaches you **distributed deep learning** from fundamentals to production deployment. You'll build real systems, not just read theory.

**By Week 8, you'll have:**
- Trained models 10-15x faster
- Built production training pipelines
- Deployed multi-node clusters
- Created your own LLM

---

## 📋 Prerequisites

### Required Knowledge
- ✅ Python programming (intermediate level)
- ✅ Basic PyTorch (nn.Module, DataLoader, optimizer.step())
- ✅ Neural network concepts
- ✅ Command line basics

### Hardware Requirements

**Minimum (Weeks 1-2):**
- 1 GPU (or Google Colab free tier)
- 8GB RAM
- 20GB disk space

**Recommended (Weeks 3-8):**
- 2-4 GPUs
- 16GB+ RAM
- 50GB disk space

**Ideal:**
- 8 GPUs or cloud access
- 32GB+ RAM
- 100GB SSD

**No GPUs?** Many exercises work on CPU or Colab!

---

## 🛠️ Environment Setup

### Option 1: Local Setup (Recommended)

```bash
# 1. Clone the repository
git clone <repo-url>
cd improved-octo-system/learning-path

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python scripts/check_setup.py

# 5. Test GPU access
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### Option 2: Google Colab

```python
# In a Colab notebook:
!git clone <repo-url>
%cd improved-octo-system/learning-path
!pip install -r requirements.txt

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Option 3: Docker

```bash
# Using provided Dockerfile
docker build -t distributed-learning .
docker run --gpus all -it distributed-learning

cd /workspace/learning-path
python scripts/check_setup.py
```

---

## 📦 Installation Steps

### 1. Install PyTorch

Visit [pytorch.org](https://pytorch.org) and select your configuration.

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

### 2. Install Framework Dependencies

```bash
cd improved-octo-system
pip install -e .  # Editable install

# Or just requirements
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Run verification script
python learning-path/scripts/check_setup.py

# Expected output:
# ✅ Python version: 3.10.x
# ✅ PyTorch version: 2.x.x
# ✅ CUDA available: True
# ✅ GPUs detected: X
# ✅ Distributed training: Ready
```

---

## 🚀 First Steps

### 1. Explore the Structure

```bash
cd learning-path
ls -la

# You'll see:
# week-1/ through week-8/  - Weekly content
# exercises/               - Practice problems
# templates/               - Code templates
# scripts/                 - Utility scripts
# notes/                   - Reference materials
# solutions/               - Reference solutions
```

### 2. Read the Course Overview

```bash
cat README.md  # Full course structure
```

### 3. Start Week 1

```bash
cd week-1
cat README.md  # Week overview
cat concepts.md  # Learn the theory
python starter_code.py  # Start coding!
```

---

## 📚 Learning Workflow

### For Each Week:

**1. Read `README.md`** - Understand goals and topics
```bash
cd week-X
cat README.md
```

**2. Study `concepts.md`** - Learn the theory
```bash
cat concepts.md
```

**3. Complete `starter_code.py`** - Fill in the blanks
```bash
# Open in your editor
code starter_code.py  # VS Code
# or
vim starter_code.py   # Vim
```

**4. Run the Lab** - Hands-on implementation
```bash
python lab_*.py
```

**5. Do Exercises** - Practice makes perfect
```bash
python exercises.py
```

**6. Verify Learning** - Check your understanding
```bash
cat checkpoints.md
python ../scripts/check_progress.py
```

**7. Compare Solutions** - Learn from reference code
```bash
cat ../solutions/week-X-solutions/solution.py
```

---

## 💻 Coding Tips

### Fill-in-the-Blank Format

Most starter code looks like this:

```python
def train_step(model, data, optimizer):
    """
    TODO: Implement a training step

    Hints:
    - Zero gradients
    - Forward pass
    - Compute loss
    - Backward pass
    - Update weights
    """
    # YOUR CODE HERE
    optimizer.zero_grad()

    # TODO: Forward pass
    output = ______(data)  # Fill in the blank

    # TODO: Compute loss
    loss = ______(output, target)

    # YOUR CODE HERE: Backward and optimize
    # ...

    return loss.item()
```

### How to Complete:

1. **Read the TODO** - Understand what's needed
2. **Check hints** - Use provided guidance
3. **Fill blanks** - Replace `______` with code
4. **Add YOUR CODE** - Write implementation
5. **Test** - Run and verify it works
6. **Compare** - Check against solution

---

## 🧪 Testing Your Code

### Run Individual Scripts

```bash
python week-1/starter_code.py
```

### Run Tests

```bash
# Test your implementation
python -m pytest week-1/test_your_code.py

# Run all tests
./scripts/run_tests.sh
```

### Verify with Solutions

```bash
# Compare your output with expected
python your_code.py > my_output.txt
python solutions/week-1/solution.py > expected_output.txt
diff my_output.txt expected_output.txt
```

---

## 📊 Track Your Progress

```bash
# Check overall progress
python scripts/check_progress.py

# Output example:
# 📊 Learning Progress
#
# Week 1: ✅ Complete
# Week 2: 🔄 In Progress (60%)
# Week 3: ⬜ Not Started
# ...
#
# Exercises: 15/50 (30%)
# Labs: 2/8 (25%)
```

---

## 🎯 Week-by-Week Timeline

### Recommended Pace: 8 weeks

| Week | Topic | Time | Complexity |
|------|-------|------|-----------|
| 1 | Foundation | 5-7 hours | ⭐ Easy |
| 2 | DDP | 6-8 hours | ⭐⭐ Medium |
| 3 | Model Parallel | 8-10 hours | ⭐⭐⭐ Hard |
| 4 | FSDP | 8-10 hours | ⭐⭐⭐ Hard |
| 5 | Optimizations | 6-8 hours | ⭐⭐ Medium |
| 6 | Best Practices | 5-7 hours | ⭐⭐ Medium |
| 7 | Deployment | 7-9 hours | ⭐⭐⭐ Hard |
| 8 | Capstone | 10-15 hours | ⭐⭐⭐⭐ Very Hard |

**Total:** 60-80 hours over 8 weeks

### Flexible Pace

**Fast Track (4 weeks):** 2 weeks of content per week
**Comfortable (8 weeks):** 1 week per week (recommended)
**Extended (16 weeks):** 0.5 weeks per week

---

## 🛠️ Useful Commands

### Environment

```bash
# Activate environment
source venv/bin/activate

# Deactivate
deactivate

# Install new package
pip install package-name

# Update requirements
pip freeze > requirements.txt
```

### GPU Utilities

```bash
# Check GPU usage
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi

# Check PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Development

```bash
# Format code
black your_script.py

# Check style
flake8 your_script.py

# Type checking
mypy your_script.py
```

---

## 📝 Note-Taking Tips

### Create Your Learning Journal

```bash
mkdir my-notes
cd my-notes

# Create weekly notes
touch week-{1..8}-notes.md

# Example structure:
# ## Week 1 Notes
#
# ### Key Concepts
# - Concept 1: ...
# - Concept 2: ...
#
# ### Code Snippets
# ```python
# # Useful pattern I learned
# ```
#
# ### Questions
# - Q: How does ...?
# - A: ...
#
# ### Next Steps
# - [ ] Review X
# - [ ] Practice Y
```

---

## 🆘 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size
batch_size = 16  # Try 8, 4, or 2
```

**2. Import Errors**
```bash
# Solution: Reinstall package
pip install --force-reinstall package-name
```

**3. Distributed Training Hangs**
```python
# Solution: Set timeout
torch.distributed.init_process_group(..., timeout=timedelta(minutes=10))
```

**4. GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Getting Help

1. Check `notes/common_errors.md`
2. Search error message in docs
3. Review solutions
4. Post in discussions
5. Debug with print statements

---

## 📚 Additional Resources

### Documentation
- Main Guide: `../docs/MASTER_GUIDE.md`
- Quick Start: `../QUICKSTART.md`
- Performance: `../PERFORMANCE_REPORT.md`

### Reference Materials
- Cheat Sheet: `notes/cheat_sheet.md`
- Glossary: `notes/glossary.md`
- Best Practices: `notes/best_practices.md`

### External Resources
- PyTorch Distributed: https://pytorch.org/tutorials/beginner/dist_overview.html
- NVIDIA Deep Learning: https://docs.nvidia.com/deeplearning/
- Papers with Code: https://paperswithcode.com/

---

## ✅ Pre-Flight Checklist

Before starting Week 1:

- [ ] Environment set up
- [ ] PyTorch installed
- [ ] GPU(s) working (or Colab ready)
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Verification script passed
- [ ] Read this guide
- [ ] Reviewed course overview
- [ ] Ready to code!

---

## 🎓 Ready to Start!

You're all set! Let's begin your journey to mastering distributed training.

```bash
cd week-1
cat README.md
python starter_code.py
```

**Welcome aboard! 🚀**

---

## 📞 Support

- **Issues:** Check `notes/common_errors.md`
- **Questions:** Review `concepts.md` for each week
- **Code Help:** See `solutions/` directory
- **Stuck:** Take a break, come back fresh!

**Remember:** Learning distributed training is challenging but incredibly rewarding. Take it one step at a time, and don't hesitate to revisit earlier material.

**You've got this! 💪**
