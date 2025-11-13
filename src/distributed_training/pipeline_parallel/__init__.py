"""
Pipeline Parallelism Module

Implements pipeline parallelism where model layers are partitioned across
devices and micro-batches are processed in a pipelined fashion.
"""

from distributed_training.pipeline_parallel.pipeline_trainer import (
    PipelineTrainer,
    PipelineStage,
)

__all__ = ["PipelineTrainer", "PipelineStage"]
