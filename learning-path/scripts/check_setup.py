#!/usr/bin/env python3
"""
Check if the learning environment is properly set up
"""

import sys


def check_python():
    """Check Python version"""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version: {version.major}.{version.minor}.{version.micro} (Need >= 3.8)")
        return False


def check_pytorch():
    """Check PyTorch installation"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: Yes")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPUs detected: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"✓ CUDA available: No (CPU only)")

        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision torchaudio")
        return False


def check_distributed():
    """Check distributed training support"""
    try:
        import torch.distributed as dist
        backends = []
        if dist.is_nccl_available():
            backends.append("NCCL")
        if dist.is_gloo_available():
            backends.append("Gloo")
        if dist.is_mpi_available():
            backends.append("MPI")

        if backends:
            print(f"✓ Distributed backends: {', '.join(backends)}")
            return True
        else:
            print("✗ No distributed backends available")
            return False
    except ImportError:
        print("✗ torch.distributed not available")
        return False


def check_dependencies():
    """Check other dependencies"""
    deps = {
        'numpy': 'numpy',
        'tqdm': 'tqdm',
        'matplotlib': 'matplotlib',
    }

    all_ok = True
    for name, import_name in deps.items():
        try:
            __import__(import_name)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} not installed")
            all_ok = False

    return all_ok


def main():
    """Run all checks"""
    print("="*60)
    print("Checking Learning Environment Setup")
    print("="*60 + "\n")

    checks = [
        ("Python Version", check_python),
        ("PyTorch", check_pytorch),
        ("Distributed Training", check_distributed),
        ("Dependencies", check_dependencies),
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        if not check_func():
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ All checks passed! Environment ready.")
        print("="*60 + "\n")
        print("Next steps:")
        print("  cd week-1")
        print("  python starter_code.py")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
