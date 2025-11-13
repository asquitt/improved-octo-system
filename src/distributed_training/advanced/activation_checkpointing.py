"""
Activation Checkpointing (Gradient Checkpointing)

Activation checkpointing trades compute for memory by not storing
intermediate activations during forward pass. Instead, they are
recomputed during backward pass.

Benefits:
---------
- Reduce memory usage by 50-80%
- Train larger models or use larger batch sizes
- Minimal impact on training speed (~20% slower)

When to Use:
------------
- Model doesn't fit in GPU memory
- Want to increase batch size
- Memory-bound training

Example:
--------
>>> model = MyLargeTransformer()
>>> enable_activation_checkpointing(model, checkpoint_every=2)
>>> # Now model uses much less memory!

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


def enable_activation_checkpointing(
    model: nn.Module,
    checkpoint_every: int = 1,
    use_reentrant: bool = True,
) -> nn.Module:
    """
    Enable activation checkpointing for a model.

    This wraps specified layers with gradient checkpointing,
    which recomputes activations during backward pass instead
    of storing them.

    Parameters:
    -----------
    model : nn.Module
        Model to enable checkpointing for
    checkpoint_every : int
        Checkpoint every N layers (1 = checkpoint all)
    use_reentrant : bool
        Use reentrant checkpointing (faster but more memory)

    Returns:
    --------
    nn.Module : Model with checkpointing enabled

    Example:
    --------
    >>> model = nn.Sequential(
    ...     nn.Linear(1000, 1000),
    ...     nn.ReLU(),
    ...     nn.Linear(1000, 1000),
    ...     nn.ReLU(),
    ... )
    >>> model = enable_activation_checkpointing(model)
    """
    # Find checkpoint-able modules
    checkpointable_layers = []

    for name, module in model.named_modules():
        # Checkpoint transformer blocks, large linear layers, etc.
        if isinstance(module, (nn.TransformerEncoderLayer,
                              nn.TransformerDecoderLayer,
                              nn.Linear)):
            if hasattr(module, 'in_features') and module.in_features > 512:
                checkpointable_layers.append((name, module))

    logger.info(f"Found {len(checkpointable_layers)} checkpointable layers")

    # Wrap layers with checkpointing
    for i, (name, module) in enumerate(checkpointable_layers):
        if i % checkpoint_every == 0:
            _wrap_module_with_checkpoint(model, name, module, use_reentrant)

    logger.info(f"Enabled activation checkpointing")

    return model


def _wrap_module_with_checkpoint(
    model: nn.Module,
    module_name: str,
    module: nn.Module,
    use_reentrant: bool,
):
    """Wrap a module with checkpoint function."""
    class CheckpointWrapper(nn.Module):
        def __init__(self, wrapped_module, use_reentrant):
            super().__init__()
            self.wrapped_module = wrapped_module
            self.use_reentrant = use_reentrant

        def forward(self, *args, **kwargs):
            # Use gradient checkpointing
            return checkpoint(
                self._forward_wrapper,
                *args,
                use_reentrant=self.use_reentrant,
                **kwargs
            )

        def _forward_wrapper(self, *args, **kwargs):
            return self.wrapped_module(*args, **kwargs)

    # Replace module with wrapped version
    parent_name = '.'.join(module_name.split('.')[:-1])
    child_name = module_name.split('.')[-1]

    if parent_name:
        parent = model.get_submodule(parent_name)
    else:
        parent = model

    wrapped = CheckpointWrapper(module, use_reentrant)
    setattr(parent, child_name, wrapped)


class ActivationCheckpointingConfig:
    """
    Configuration for activation checkpointing.

    Use this for fine-grained control over checkpointing behavior.

    Attributes:
    -----------
    enabled : bool
        Enable/disable checkpointing
    checkpoint_every_n_layers : int
        Checkpoint every N layers
    checkpoint_transformer_blocks : bool
        Checkpoint transformer encoder/decoder blocks
    checkpoint_large_linear : bool
        Checkpoint large linear layers (>512 features)
    use_reentrant : bool
        Use reentrant checkpointing
    """

    def __init__(
        self,
        enabled: bool = True,
        checkpoint_every_n_layers: int = 1,
        checkpoint_transformer_blocks: bool = True,
        checkpoint_large_linear: bool = True,
        use_reentrant: bool = True,
    ):
        """Initialize checkpointing configuration."""
        self.enabled = enabled
        self.checkpoint_every_n_layers = checkpoint_every_n_layers
        self.checkpoint_transformer_blocks = checkpoint_transformer_blocks
        self.checkpoint_large_linear = checkpoint_large_linear
        self.use_reentrant = use_reentrant


def apply_activation_checkpointing(
    model: nn.Module,
    config: ActivationCheckpointingConfig,
) -> nn.Module:
    """
    Apply activation checkpointing based on configuration.

    Parameters:
    -----------
    model : nn.Module
        Model to apply checkpointing to
    config : ActivationCheckpointingConfig
        Checkpointing configuration

    Returns:
    --------
    nn.Module : Model with checkpointing applied

    Example:
    --------
    >>> config = ActivationCheckpointingConfig(
    ...     enabled=True,
    ...     checkpoint_every_n_layers=2,
    ...     checkpoint_transformer_blocks=True
    ... )
    >>> model = apply_activation_checkpointing(model, config)
    """
    if not config.enabled:
        return model

    checkpointable = []

    for name, module in model.named_modules():
        should_checkpoint = False

        # Transformer blocks
        if config.checkpoint_transformer_blocks:
            if isinstance(module, (nn.TransformerEncoderLayer,
                                  nn.TransformerDecoderLayer)):
                should_checkpoint = True

        # Large linear layers
        if config.checkpoint_large_linear:
            if isinstance(module, nn.Linear):
                if hasattr(module, 'in_features') and module.in_features > 512:
                    should_checkpoint = True

        if should_checkpoint:
            checkpointable.append((name, module))

    # Apply checkpointing
    for i, (name, module) in enumerate(checkpointable):
        if i % config.checkpoint_every_n_layers == 0:
            _wrap_module_with_checkpoint(model, name, module, config.use_reentrant)

    logger.info(
        f"Applied activation checkpointing to {len(checkpointable)} layers"
    )

    return model


def estimate_memory_savings(
    model: nn.Module,
    input_shape: tuple,
    checkpoint_enabled: bool = True,
) -> dict:
    """
    Estimate memory savings from activation checkpointing.

    Parameters:
    -----------
    model : nn.Module
        Model to analyze
    input_shape : tuple
        Input tensor shape
    checkpoint_enabled : bool
        Whether checkpointing is enabled

    Returns:
    --------
    dict : Memory estimates

    Example:
    --------
    >>> model = MyModel()
    >>> savings = estimate_memory_savings(
    ...     model,
    ...     input_shape=(3, 224, 224)
    ... )
    >>> print(f"Memory saved: {savings['savings_percent']:.1f}%")
    """
    # Count activations
    num_activations = 0

    def activation_hook(module, input, output):
        nonlocal num_activations
        if isinstance(output, torch.Tensor):
            num_activations += output.numel() * output.element_size()

    # Register hooks
    hooks = []
    for module in model.modules():
        hook = module.register_forward_hook(activation_hook)
        hooks.append(hook)

    # Forward pass
    dummy_input = torch.randn(1, *input_shape)
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Calculate memory
    activation_memory_mb = num_activations / (1024 ** 2)

    if checkpoint_enabled:
        # Typically 60-80% reduction in activation memory
        savings_percent = 70
        saved_memory_mb = activation_memory_mb * (savings_percent / 100)
    else:
        savings_percent = 0
        saved_memory_mb = 0

    return {
        "total_activation_memory_mb": activation_memory_mb,
        "saved_memory_mb": saved_memory_mb,
        "savings_percent": savings_percent,
        "final_memory_mb": activation_memory_mb - saved_memory_mb,
    }
