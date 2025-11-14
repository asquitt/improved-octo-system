#!/usr/bin/env python3
"""Template for model definition"""

import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers
        pass

    def forward(self, x):
        # Forward pass
        return x

    def compute_loss(self, outputs, targets):
        # Custom loss computation
        pass
