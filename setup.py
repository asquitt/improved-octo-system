"""
Setup script for Distributed Training Framework.

This allows the package to be installed with pip install -e .
for development, making imports easier across modules.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="distributed-training",
    version="0.1.0",
    author="Your Name",
    description="Production-ready distributed training framework for large ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/distributed-training",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "deepspeed>=0.12.0",
        "ray[train]>=2.9.0",
        "accelerate>=0.25.0",
        "tensorboard>=2.15.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
)
