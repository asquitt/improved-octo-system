#!/usr/bin/env python3
"""Template for configuration"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model
    model_name: str = 'resnet50'
    hidden_size: int = 768
    num_layers: int = 12

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4

    # Distributed
    world_size: int = 4
    backend: str = 'nccl'

    # Optimization
    use_amp: bool = True
    use_compile: bool = True
    gradient_clip: float = 1.0
