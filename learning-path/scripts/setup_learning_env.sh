#!/bin/bash
#
# Setup script for the learning path environment
#
# Usage: ./scripts/setup_learning_env.sh

set -e

echo "🚀 Setting up Distributed Training Learning Environment"
echo "=========================================================="

# Check Python version
echo -e "\n📋 Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "  ✓ Python $PYTHON_VERSION (OK)"
else
    echo "  ✗ Python $PYTHON_VERSION (Need >= 3.8)"
    exit 1
fi

# Create virtual environment
echo -e "\n🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  ℹ Virtual environment already exists"
else
    python -m venv venv
    echo "  ✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n🎯 Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo "  ✓ Virtual environment activated"

# Upgrade pip
echo -e "\n📦 Upgrading pip..."
pip install --upgrade pip > /dev/null
echo "  ✓ pip upgraded"

# Install PyTorch
echo -e "\n🔥 Installing PyTorch..."
echo "  Detecting CUDA..."

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "  CUDA detected: $CUDA_VERSION"

    if [[ "$CUDA_VERSION" == 12.* ]]; then
        echo "  Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == 11.* ]]; then
        echo "  Installing PyTorch with CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  Installing PyTorch (CPU version)..."
        pip install torch torchvision torchaudio
    fi
else
    echo "  No GPU detected, installing CPU version..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies
echo -e "\n📚 Installing additional dependencies..."
pip install -r requirements.txt > /dev/null
echo "  ✓ Dependencies installed"

# Create data directory
echo -e "\n📁 Creating data directory..."
mkdir -p data
echo "  ✓ Data directory created"

# Verify installation
echo -e "\n✅ Verifying installation..."
python scripts/check_setup.py

echo -e "\n=========================================================="
echo "🎉 Setup complete!"
echo "=========================================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start learning:"
echo "     cd week-1"
echo "     python starter_code.py"
echo ""
echo "Happy learning! 🚀"
