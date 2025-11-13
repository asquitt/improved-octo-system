# Distributed Training Framework - Docker Image
# Production-ready container for distributed training

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create directories
RUN mkdir -p /workspace/checkpoints /workspace/data /workspace/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NCCL_DEBUG=INFO

# Default command
CMD ["/bin/bash"]
