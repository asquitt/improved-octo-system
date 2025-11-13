"""
torch.compile Integration for Distributed Training

This module provides integration of torch.compile with distributed training strategies.
torch.compile (PyTorch 2.0+) provides significant performance improvements through:
- Automatic kernel fusion
- Graph optimization
- Backend-specific code generation

Performance Benefits (Research-Backed):
- IBM Research 2024: 4,550 tokens/sec/GPU on A100 (Granite 7B)
- TorchTitan 2024: 30-65% throughput improvement
- Better with FSDP: Up to 1.5x speedup combined with FP8

Key Features:
- Automatic model compilation
- Distributed training compatibility
- Dynamic shape handling
- Reduced Python overhead
- Backend selection (inductor, cudagraphs, etc.)

Usage:
    >>> from distributed_training.advanced.compile_trainer import CompileConfig, compile_model
    >>>
    >>> # Basic compilation
    >>> compiled_model = compile_model(model)
    >>>
    >>> # With custom config
    >>> config = CompileConfig(backend='inductor', mode='max-autotune')
    >>> compiled_model = compile_model(model, config)
    >>>
    >>> # Integrated with FSDP
    >>> from distributed_training.advanced.fsdp_trainer import FSDPTrainer
    >>> trainer = FSDPTrainer(model, compile=True)

References:
- PyTorch 2.0 torch.compile: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- TorchTitan: https://arxiv.org/abs/2410.06511
- IBM Research 2024: Fastest training on A100 GPUs
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class CompileConfig:
    """
    Configuration for torch.compile compilation.

    Attributes:
        backend: Compiler backend ('inductor', 'aot_eager', 'cudagraphs', etc.)
        mode: Optimization mode ('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs')
        fullgraph: Require the graph to be captured entirely (no graph breaks)
        dynamic: Enable dynamic shapes
        options: Additional backend-specific options
        disable: Whether to disable compilation (for debugging)

    Modes Explained:
    - 'default': Balanced compile time and runtime performance
    - 'reduce-overhead': Minimize Python overhead (CUDA graphs)
    - 'max-autotune': Aggressive optimization, longer compile time
    - 'max-autotune-no-cudagraphs': max-autotune without CUDA graphs

    Backend Options:
    - 'inductor': Default, production backend (TorchInductor)
    - 'cudagraphs': CUDA graphs for reduced overhead
    - 'aot_eager': AOT Autograd with eager execution (debugging)
    """
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: Optional[bool] = None
    options: Optional[Dict[str, Any]] = None
    disable: bool = False


def compile_model(
    model: nn.Module,
    config: Optional[CompileConfig] = None,
    **compile_kwargs
) -> nn.Module:
    """
    Compile a PyTorch model using torch.compile.

    This function wraps torch.compile with distributed training best practices:
    - Compiles before wrapping in DDP/FSDP
    - Handles dynamic shapes appropriately
    - Provides sensible defaults for different use cases

    Args:
        model: PyTorch model to compile
        config: CompileConfig object with compilation settings
        **compile_kwargs: Additional arguments passed to torch.compile

    Returns:
        Compiled model

    Example:
        >>> # Basic compilation
        >>> model = MyModel()
        >>> compiled_model = compile_model(model)
        >>>
        >>> # Maximum optimization
        >>> config = CompileConfig(mode='max-autotune')
        >>> compiled_model = compile_model(model, config)
        >>>
        >>> # With FSDP (compile before FSDP wrap)
        >>> compiled_model = compile_model(model)
        >>> fsdp_model = FSDP(compiled_model, ...)

    Note:
        - Always compile BEFORE wrapping with DDP/FSDP
        - First forward pass will trigger compilation (slow)
        - Subsequent passes will be significantly faster
        - Dynamic shapes may reduce optimization effectiveness
    """
    if config is None:
        config = CompileConfig()

    if config.disable:
        logger.info("torch.compile disabled, returning original model")
        return model

    # Check PyTorch version
    pytorch_version = torch.__version__.split('.')
    major, minor = int(pytorch_version[0]), int(pytorch_version[1])

    if major < 2:
        logger.warning(
            f"torch.compile requires PyTorch 2.0+, current version: {torch.__version__}. "
            "Returning uncompiled model."
        )
        return model

    # Prepare compile arguments
    compile_args = {
        'backend': config.backend,
        'mode': config.mode,
        'fullgraph': config.fullgraph,
    }

    if config.dynamic is not None:
        compile_args['dynamic'] = config.dynamic

    if config.options is not None:
        compile_args['options'] = config.options

    # Override with any additional kwargs
    compile_args.update(compile_kwargs)

    logger.info(f"Compiling model with torch.compile")
    logger.info(f"  Backend: {compile_args['backend']}")
    logger.info(f"  Mode: {compile_args['mode']}")
    logger.info(f"  Fullgraph: {compile_args['fullgraph']}")

    try:
        compiled_model = torch.compile(model, **compile_args)
        logger.info("✓ Model compilation successful")
        logger.info("  Note: First forward pass will trigger compilation (may be slow)")
        return compiled_model
    except Exception as e:
        logger.error(f"Failed to compile model: {e}")
        logger.warning("Returning uncompiled model")
        return model


class CompiledTrainerMixin:
    """
    Mixin class to add torch.compile support to trainers.

    This mixin provides:
    - Automatic model compilation
    - Compilation timing measurement
    - Fallback to eager mode on compilation failure

    Usage:
        >>> class MyTrainer(CompiledTrainerMixin, BaseTrainer):
        >>>     def __init__(self, model, compile_config=None, **kwargs):
        >>>         self.setup_compilation(model, compile_config)
        >>>         super().__init__(model, **kwargs)
    """

    def setup_compilation(
        self,
        model: nn.Module,
        compile_config: Optional[CompileConfig] = None,
        compile_before_wrap: bool = True
    ) -> nn.Module:
        """
        Setup torch.compile for the model.

        Args:
            model: Model to compile
            compile_config: Compilation configuration
            compile_before_wrap: Whether to compile before DDP/FSDP wrapping

        Returns:
            Compiled (or original) model
        """
        if compile_config is None or compile_config.disable:
            logger.info("Compilation disabled")
            return model

        if compile_before_wrap:
            logger.info("Compiling model before distributed wrapping (recommended)")
            return compile_model(model, compile_config)
        else:
            logger.warning(
                "Compiling after distributed wrapping may reduce optimization effectiveness"
            )
            return model

    def compile_after_wrap(
        self,
        model: nn.Module,
        compile_config: Optional[CompileConfig] = None
    ) -> nn.Module:
        """
        Compile model after DDP/FSDP wrapping.

        Note: Compiling before wrapping is generally recommended.

        Args:
            model: Wrapped model to compile
            compile_config: Compilation configuration

        Returns:
            Compiled model
        """
        if compile_config is None or compile_config.disable:
            return model

        logger.warning("Compiling after wrapping - may reduce effectiveness")
        return compile_model(model, compile_config)


def create_compiled_training_step(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    compile_config: Optional[CompileConfig] = None
) -> Callable:
    """
    Create a compiled training step function.

    This compiles the entire training step (forward + backward + optimizer step)
    for maximum performance. Useful for small models or when using reduce-overhead mode.

    Args:
        model: PyTorch model
        loss_fn: Loss function
        optimizer: Optimizer
        compile_config: Compilation configuration

    Returns:
        Compiled training step function

    Example:
        >>> model = MyModel()
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>>
        >>> config = CompileConfig(mode='reduce-overhead')
        >>> train_step = create_compiled_training_step(
        >>>     model, loss_fn, optimizer, config
        >>> )
        >>>
        >>> for batch in dataloader:
        >>>     loss = train_step(batch['input'], batch['target'])

    Note:
        This is most effective with reduce-overhead mode + CUDA graphs.
        May not work well with dynamic shapes or complex training loops.
    """
    if compile_config is None:
        compile_config = CompileConfig()

    def training_step(data, target):
        """Single training step"""
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        return loss

    if config.disable:
        logger.info("Returning uncompiled training step")
        return training_step

    logger.info("Compiling training step function")
    logger.info(f"  Mode: {compile_config.mode}")
    logger.info("  Note: This compiles forward + backward + optimizer step")

    try:
        compiled_step = torch.compile(
            training_step,
            backend=compile_config.backend,
            mode=compile_config.mode,
            fullgraph=compile_config.fullgraph
        )
        logger.info("✓ Training step compilation successful")
        return compiled_step
    except Exception as e:
        logger.error(f"Failed to compile training step: {e}")
        logger.warning("Returning uncompiled training step")
        return training_step


class CompilationProfiler:
    """
    Profiler for tracking torch.compile performance benefits.

    This class helps measure:
    - Compilation time (first run)
    - Speedup vs eager mode
    - Memory usage comparison

    Usage:
        >>> profiler = CompilationProfiler()
        >>>
        >>> # Profile eager mode
        >>> with profiler.profile('eager'):
        >>>     output = model(data)
        >>>
        >>> # Compile and profile
        >>> compiled_model = compile_model(model)
        >>> with profiler.profile('compiled'):
        >>>     output = compiled_model(data)
        >>>
        >>> # Get results
        >>> results = profiler.get_results()
        >>> print(f"Speedup: {results['speedup']:.2f}x")
    """

    def __init__(self):
        self.times = {}
        self.memory = {}

    def profile(self, name: str):
        """Context manager for profiling a code section"""
        import time
        import contextlib

        @contextlib.contextmanager
        def _profile():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            yield

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            end_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

            self.times[name] = end_time - start_time
            self.memory[name] = end_mem - start_mem

        return _profile()

    def get_results(self) -> Dict[str, Any]:
        """Get profiling results"""
        results = {
            'times': self.times,
            'memory': self.memory
        }

        if 'eager' in self.times and 'compiled' in self.times:
            results['speedup'] = self.times['eager'] / self.times['compiled']
            results['time_saved_ms'] = (self.times['eager'] - self.times['compiled']) * 1000

        if 'eager' in self.memory and 'compiled' in self.memory:
            results['memory_change_mb'] = (
                (self.memory['compiled'] - self.memory['eager']) / (1024 ** 2)
            )

        return results

    def print_results(self):
        """Print profiling results in a readable format"""
        results = self.get_results()

        print("\n" + "=" * 60)
        print("torch.compile Performance Profiling Results")
        print("=" * 60)

        for name, time_val in results['times'].items():
            print(f"\n{name.upper()} mode:")
            print(f"  Time: {time_val*1000:.2f} ms")
            if name in results['memory']:
                print(f"  Memory: {results['memory'][name]/(1024**2):.2f} MB")

        if 'speedup' in results:
            print(f"\nPERFORMANCE IMPROVEMENT:")
            print(f"  Speedup: {results['speedup']:.2f}x")
            print(f"  Time saved: {results['time_saved_ms']:.2f} ms")

        if 'memory_change_mb' in results:
            mem_change = results['memory_change_mb']
            sign = '+' if mem_change > 0 else ''
            print(f"  Memory change: {sign}{mem_change:.2f} MB")

        print("=" * 60 + "\n")


# Example configuration presets
COMPILE_PRESETS = {
    'development': CompileConfig(
        backend='inductor',
        mode='default',
        fullgraph=False,
        dynamic=True
    ),
    'production': CompileConfig(
        backend='inductor',
        mode='max-autotune',
        fullgraph=False,
        dynamic=False
    ),
    'low_latency': CompileConfig(
        backend='cudagraphs',
        mode='reduce-overhead',
        fullgraph=True,
        dynamic=False
    ),
    'debug': CompileConfig(
        backend='aot_eager',
        mode='default',
        fullgraph=False,
        dynamic=True
    ),
    'disabled': CompileConfig(
        disable=True
    )
}


def get_compile_preset(preset_name: str) -> CompileConfig:
    """
    Get a predefined compilation configuration.

    Args:
        preset_name: Name of preset ('development', 'production', 'low_latency', 'debug', 'disabled')

    Returns:
        CompileConfig object

    Presets:
    - 'development': Fast compile, supports dynamic shapes
    - 'production': Maximum optimization, longer compile time
    - 'low_latency': CUDA graphs, minimal overhead
    - 'debug': Eager mode, easier debugging
    - 'disabled': No compilation
    """
    if preset_name not in COMPILE_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(COMPILE_PRESETS.keys())}"
        )

    return COMPILE_PRESETS[preset_name]
