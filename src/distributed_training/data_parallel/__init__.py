"""
Data Parallelism Module

Implements PyTorch DistributedDataParallel (DDP) for multi-GPU training.
Data parallelism replicates the model on each GPU and distributes different
batches of data to each replica.
"""

from distributed_training.data_parallel.ddp_trainer import DDPTrainer

__all__ = ["DDPTrainer"]
