"""
Distributed Training Framework

A comprehensive framework for training large ML models using various
parallelism strategies.
"""

__version__ = "0.1.0"

from distributed_training.data_parallel import DDPTrainer
from distributed_training.model_parallel import TensorParallelModel
from distributed_training.pipeline_parallel import PipelineTrainer

__all__ = [
    "DDPTrainer",
    "TensorParallelModel",
    "PipelineTrainer",
]
