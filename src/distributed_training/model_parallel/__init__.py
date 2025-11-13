"""
Model Parallelism Module

Implements tensor parallelism for models that don't fit in single GPU memory.
Model parallelism splits the model itself across multiple GPUs.
"""

from distributed_training.model_parallel.tensor_parallel import (
    TensorParallelModel,
    ColumnParallelLinear,
    RowParallelLinear,
)

__all__ = [
    "TensorParallelModel",
    "ColumnParallelLinear",
    "RowParallelLinear",
]
