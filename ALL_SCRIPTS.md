# 📜 ALL SCRIPTS TO RUN - COMPLETE REFERENCE

This document contains **every script you need** to use the framework, organized by purpose.

---

## 🚀 INSTALLATION SCRIPTS

### Install Framework
```bash
# Clone repository
git clone <repository-url>
cd improved-octo-system

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
python -c "import distributed_training; print('✓ Success!')"
```

### Install with Conda
```bash
# Create environment
conda create -n dist-training python=3.10
conda activate dist-training

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Docker Installation
```bash
# Build image
docker build -t distributed-training .

# Run container
docker run --gpus all -it distributed-training bash

# Or use docker-compose
docker-compose up -d
```

---

## ⚙️ CONFIGURATION SCRIPTS

### Generate Default Configs
```bash
# Create all default configuration files
python -m distributed_training.cli.train_cli --create-configs

# This creates:
# - configs/single_gpu.yaml
# - configs/multi_gpu_ddp.yaml
# - configs/deepspeed_zero2.yaml
# - configs/deepspeed_zero3_offload.yaml
# - configs/fsdp.yaml
```

### View Config
```bash
# Print config contents
cat configs/multi_gpu_ddp.yaml
```

### Custom Config
```yaml
# Create custom config: configs/my_config.yaml
num_epochs: 20
batch_size: 64
learning_rate: 0.001
strategy: ddp
num_gpus: 4
mixed_precision: true
```

---

## 🏃 TRAINING SCRIPTS

### Single GPU Training
```bash
# Basic training
python examples/01_data_parallel_simple.py

# Or with torchrun
torchrun --nproc_per_node=1 examples/01_data_parallel_simple.py
```

### Multi-GPU Training (DDP)
```bash
# 2 GPUs
torchrun --nproc_per_node=2 examples/01_data_parallel_simple.py

# 4 GPUs
torchrun --nproc_per_node=4 examples/01_data_parallel_simple.py

# 8 GPUs
torchrun --nproc_per_node=8 examples/01_data_parallel_simple.py
```

### Model Parallel Training
```bash
# Run on 4 GPUs
torchrun --nproc_per_node=4 examples/02_model_parallel_large_model.py
```

### Pipeline Parallel Training
```bash
# Run on 4 GPUs
torchrun --nproc_per_node=4 examples/03_pipeline_parallel_transformer.py
```

### Complete Demo
```bash
# All features demo
torchrun --nproc_per_node=2 examples/04_complete_training_demo.py
```

### Multi-Node Training
```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py

# Node 1
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py
```

### With CLI
```bash
# Train with config file
python -m distributed_training.cli.train_cli --config configs/ddp.yaml

# Override parameters
python -m distributed_training.cli.train_cli \
    --config configs/ddp.yaml \
    --num-epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001

# Different strategy
python -m distributed_training.cli.train_cli \
    --config configs/fsdp.yaml \
    --num-gpus 4
```

---

## 🔍 OPTIMIZATION SCRIPTS

### Find Optimal Batch Size
```python
# find_batch_size.py
import torch
from distributed_training.optimization import find_optimal_batch_size
from my_model import MyModel

model = MyModel()

optimal_bs = find_optimal_batch_size(
    model=model,
    input_shape=(3, 224, 224),
    max_batch_size=512,
    min_batch_size=1
)

print(f"Optimal batch size: {optimal_bs}")
```

```bash
# Run script
python find_batch_size.py
```

### Find Optimal Learning Rate
```python
# find_lr.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from distributed_training.optimization import LRFinder
from my_model import MyModel

# Setup
X = torch.randn(1000, 3, 224, 224)
y = torch.randint(0, 10, (1000,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=32)

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# Find LR
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10)
lr_finder.plot(save_path="lr_finder.png")

optimal_lr = lr_finder.get_best_lr()
print(f"Optimal learning rate: {optimal_lr:.2e}")
```

```bash
# Run script
python find_lr.py

# View plot
open lr_finder.png  # macOS
xdg-open lr_finder.png  # Linux
```

### CLI Optimization
```bash
# Find batch size
python -m distributed_training.cli.train_cli --find-batch-size

# Find learning rate
python -m distributed_training.cli.train_cli \
    --config configs/ddp.yaml \
    --find-lr
```

---

## 📊 BENCHMARKING SCRIPTS

### Simple Benchmark
```python
# benchmark.py
from distributed_training.benchmarking import BenchmarkRunner
from my_model import MyModel
from torch.utils.data import TensorDataset
import torch

# Setup
model = MyModel()
X = torch.randn(1000, 3, 32, 32)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)

# Create benchmark
benchmark = BenchmarkRunner()

# Add configurations to compare
benchmark.add_config("DDP-2GPU", strategy="ddp", num_gpus=2, batch_size=32)
benchmark.add_config("DDP-4GPU", strategy="ddp", num_gpus=4, batch_size=64)

# Run benchmarks
results = benchmark.run(model, dataset, num_iterations=100)

# Show results
benchmark.print_results()
benchmark.plot_comparison(save_path="benchmark.png")
benchmark.save_results("benchmark_results.json")
```

```bash
# Run benchmark
python benchmark.py

# View results
cat benchmark_results.json
open benchmark.png
```

---

## 🧪 TESTING SCRIPTS

### Run All Tests
```bash
# Run all unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/distributed_training --cov-report=html

# View coverage
open htmlcov/index.html
```

### Run Specific Tests
```bash
# Data parallel tests
pytest tests/test_data_parallel.py -v

# Model parallel tests
pytest tests/test_model_parallel.py -v

# Checkpointing tests
pytest tests/test_checkpointing.py -v

# Profiling tests
pytest tests/test_profiling.py -v

# Data loading tests
pytest tests/test_data_loading.py -v
```

### Run by Category
```bash
# Unit tests only
pytest tests/ -v -m unit

# Integration tests
pytest tests/ -v -m integration

# Distributed tests (requires multiple GPUs)
pytest tests/ -v -m distributed

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Code Quality Checks
```bash
# Format code
black src/ tests/ examples/

# Check formatting
black --check src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/ --max-line-length=127

# Type checking
mypy src/
```

---

## 🐳 DOCKER SCRIPTS

### Build and Run
```bash
# Build image
docker build -t distributed-training .

# Run container with GPU
docker run --gpus all -it distributed-training bash

# Run specific example
docker run --gpus all distributed-training \
    python examples/01_data_parallel_simple.py
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access trainer container
docker-compose exec trainer bash

# Stop all services
docker-compose down
```

### With Volume Mounts
```bash
# Mount local data
docker run --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -it distributed-training bash
```

---

## 📈 MONITORING SCRIPTS

### GPU Monitoring
```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Detailed GPU info
nvidia-smi -l 1

# GPU metrics to file
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > gpu_metrics.csv
```

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir=./logs --port=6006

# Access at http://localhost:6006

# Or with docker-compose
docker-compose up tensorboard
```

### System Monitoring
```bash
# CPU and memory
htop

# Network
iftop

# Disk I/O
iotop
```

---

## 📊 VISUALIZATION SCRIPTS

### Generate Training Plots
```python
# visualize.py
from distributed_training.visualization import (
    TrainingVisualizer,
    plot_training_curves,
    plot_gpu_utilization,
    create_performance_dashboard
)

# Compare multiple runs
viz = TrainingVisualizer()
viz.add_run("DDP", losses=[0.5, 0.3, 0.2], accuracies=[0.8, 0.9, 0.95])
viz.add_run("FSDP", losses=[0.5, 0.3, 0.19], accuracies=[0.8, 0.91, 0.96])
viz.plot_comparison(save_path="comparison.png")

# Plot single run
plot_training_curves(
    train_losses=[0.5, 0.3, 0.2],
    val_losses=[0.6, 0.4, 0.25],
    train_accs=[0.8, 0.9, 0.95],
    val_accs=[0.75, 0.85, 0.92],
    save_path="training_curves.png"
)

# GPU utilization
plot_gpu_utilization(
    gpu_utils=[75, 80, 82, 85, 88, 90],
    save_path="gpu_util.png"
)

# Full dashboard
metrics = {
    "train_losses": [0.5, 0.3, 0.2],
    "val_losses": [0.6, 0.4, 0.25],
    "gpu_util": [75, 80, 82],
    "throughput": [100, 120, 125],
    "memory_usage": [8, 8.5, 9],
}
create_performance_dashboard(metrics, save_path="dashboard.png")
```

```bash
# Run visualization
python visualize.py

# View plots
open *.png
```

---

## 🔧 UTILITY SCRIPTS

### Check System Info
```bash
# Python version
python --version

# PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# CUDA version
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"

# GPU count
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# GPU names
nvidia-smi --query-gpu=name --format=csv,noheader

# Check distributed
python -c "import torch.distributed as dist; print('Distributed available:', dist.is_available())"

# Check NCCL
python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
```

### Environment Setup
```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set master port
export MASTER_PORT=29500

# Set NCCL debug
export NCCL_DEBUG=INFO

# Set NCCL interface
export NCCL_SOCKET_IFNAME=eth0

# Disable CUDA cache
export CUDA_CACHE_DISABLE=1
```

### Cleanup
```bash
# Kill all Python processes
pkill -9 python

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Remove checkpoints
rm -rf checkpoints/*

# Remove logs
rm -rf logs/*

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

## 📦 DEPLOYMENT SCRIPTS

### Cloud Deployment (AWS)

#### Launch EC2 Instance
```bash
# Launch p3.8xlarge with 4 V100 GPUs
aws ec2 run-instances \
    --image-id ami-xxxxxxxxx \
    --instance-type p3.8xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxxxxxxx
```

#### Deploy with Docker
```bash
# SSH to instance
ssh -i your-key.pem ubuntu@ec2-xx-xxx-xxx-xx.compute.amazonaws.com

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone and run
git clone <repository-url>
cd distributed-training
docker build -t distributed-training .
docker run --gpus all -it distributed-training
```

### Kubernetes Deployment
```yaml
# k8s-training-job.yaml
apiVersion: v1
kind: Pod
metadata:
  name: distributed-training
spec:
  containers:
  - name: trainer
    image: distributed-training:latest
    resources:
      limits:
        nvidia.com/gpu: 4
    command: ["torchrun"]
    args: ["--nproc_per_node=4", "train.py"]
    volumeMounts:
    - name: data
      mountPath: /workspace/data
    - name: checkpoints
      mountPath: /workspace/checkpoints
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: training-data-pvc
  - name: checkpoints
    persistentVolumeClaim:
      claimName: training-checkpoints-pvc
```

```bash
# Deploy
kubectl apply -f k8s-training-job.yaml

# Monitor
kubectl logs -f distributed-training

# Get shell
kubectl exec -it distributed-training -- bash
```

### SLURM Deployment
```bash
#!/bin/bash
#SBATCH --job-name=distributed-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=training_%j.out

# Load modules
module load python/3.10
module load cuda/12.1

# Setup environment
source venv/bin/activate

# Run training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$SLURM_NODELIST \
    --master_port=29500 \
    train.py
```

```bash
# Submit job
sbatch slurm_train.sh

# Check status
squeue -u $USER

# View output
tail -f training_*.out
```

---

## 🆘 TROUBLESHOOTING SCRIPTS

### Debug Script
```bash
# debug.sh
#!/bin/bash

echo "=== System Info ==="
python --version
nvidia-smi

echo "=== PyTorch Info ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

echo "=== Test Import ==="
python -c "import distributed_training; print('✓ Import successful')"

echo "=== Run Simple Test ==="
pytest tests/test_data_parallel.py::TestDDPTrainer::test_ddp_trainer_creation -v
```

### Fix Common Issues
```bash
# Issue: Port already in use
export MASTER_PORT=29501

# Issue: CUDA out of memory
python -c "import torch; torch.cuda.empty_cache()"

# Issue: Import errors
pip install -e .

# Issue: Permission denied
chmod +x scripts/*.sh

# Issue: Stale processes
pkill -9 python
```

---

## 📝 QUICK REFERENCE

### Essential Commands
```bash
# Install
pip install -e .

# Create configs
python -m distributed_training.cli.train_cli --create-configs

# Train (2 GPUs)
torchrun --nproc_per_node=2 examples/01_data_parallel_simple.py

# Test
pytest tests/ -v

# Monitor
watch -n 1 nvidia-smi

# Visualize
tensorboard --logdir=./logs
```

### Common Workflows

**Single GPU Training:**
```bash
python examples/01_data_parallel_simple.py
```

**Multi-GPU Training:**
```bash
torchrun --nproc_per_node=4 examples/01_data_parallel_simple.py
```

**Find Optimal Settings:**
```bash
python find_batch_size.py
python find_lr.py
```

**Benchmark Strategies:**
```bash
python benchmark.py
```

**Deploy with Docker:**
```bash
docker-compose up -d
```

---

**All scripts ready to use!** 🚀

Start with the Quick Reference and explore from there.
