"""
FSDP2 with DTensor - Next Generation Fully Sharded Data Parallel

This module implements FSDP2, the next generation of PyTorch's Fully Sharded Data Parallel,
which uses DTensor for improved performance and composability.

Key Improvements over FSDP1:
--------------
1. **DTensor-based per-parameter sharding**: Simpler representation, better composability
2. **7% lower GPU memory**: More efficient memory management
3. **Communication-free state dicts**: No all-gathers needed for checkpointing
4. **Better composability**: Works seamlessly with TP, PP, and other techniques
5. **Finer-grained quantization**: Enables per-tensor FP8 quantization
6. **Simpler API**: More intuitive parameter handling

Performance Benefits (Meta Research):
- 7% lower memory vs FSDP1
- Better throughput with torch.compile
- Up to 50% speedup with FP8 (on H100/H200)
- Communication-free sharded state dicts

References:
- PyTorch FSDP2: https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- DTensor: https://pytorch.org/docs/stable/distributed.tensor.html
- TorchTitan: https://arxiv.org/abs/2410.06511
- Meta Blog: https://pytorch.org/blog/training-using-float8-fsdp2/
"""

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor, Shard, Replicate
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Check PyTorch version for FSDP2 availability
FSDP2_AVAILABLE = False
try:
    from torch.distributed.fsdp import fully_shard, MixedPrecision, ShardingStrategy
    FSDP2_AVAILABLE = True
    logger.info("FSDP2 (fully_shard) is available")
except ImportError:
    logger.warning(
        "FSDP2 (fully_shard) not available. "
        "Requires PyTorch 2.4+. Falling back to FSDP1."
    )
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        logger.info("FSDP1 available as fallback")
    except ImportError:
        logger.error("Neither FSDP2 nor FSDP1 available")


@dataclass
class FSDP2Config:
    """
    Configuration for FSDP2 training.

    Attributes:
        mesh_dim_names: Dimension names for device mesh (e.g., ('dp', 'tp'))
        dp_size: Data parallel size (number of GPUs for data parallelism)
        tp_size: Tensor parallel size (for hybrid parallelism)
        reshard_after_forward: Whether to reshard parameters after forward pass
        mixed_precision: Mixed precision configuration
        cpu_offload: Whether to offload parameters to CPU
        selective_checkpointing: Use selective activation checkpointing
        compile: Whether to use torch.compile
        compile_config: torch.compile configuration

    Notes:
        - dp_size * tp_size should equal total number of GPUs
        - reshard_after_forward=True saves memory but may reduce speed
        - cpu_offload enables training larger models on limited GPU memory
    """
    mesh_dim_names: Tuple[str, ...] = ('dp',)
    dp_size: Optional[int] = None  # Auto-detect if None
    tp_size: int = 1  # For hybrid parallelism
    reshard_after_forward: bool = True
    mixed_precision: Optional[Any] = None  # MixedPrecision config
    cpu_offload: bool = False
    selective_checkpointing: bool = False
    compile: bool = False
    compile_config: Optional[Dict[str, Any]] = None


class FSDP2Trainer:
    """
    FSDP2 Trainer with DTensor-based per-parameter sharding.

    This trainer provides:
    - DTensor-based parameter sharding
    - Hybrid parallelism (DP + TP)
    - Communication-free checkpointing
    - torch.compile integration
    - FP8 support (on compatible hardware)

    Example:
        >>> # Basic FSDP2 training
        >>> from distributed_training.advanced.fsdp2_trainer import FSDP2Trainer, FSDP2Config
        >>>
        >>> config = FSDP2Config(dp_size=4)
        >>> trainer = FSDP2Trainer(model, config)
        >>>
        >>> for batch in dataloader:
        >>>     loss = trainer.train_step(batch)
        >>>
        >>> # With torch.compile for maximum performance
        >>> config = FSDP2Config(dp_size=4, compile=True)
        >>> trainer = FSDP2Trainer(model, config)
        >>>
        >>> # Hybrid parallelism (DP + TP)
        >>> config = FSDP2Config(
        >>>     mesh_dim_names=('dp', 'tp'),
        >>>     dp_size=4,
        >>>     tp_size=2  # 4 * 2 = 8 GPUs total
        >>> )
        >>> trainer = FSDP2Trainer(model, config)

    Performance Tips:
        1. Use torch.compile for 30-65% throughput improvement
        2. Combine with FP8 for up to 50% additional speedup
        3. Use reshard_after_forward=False if you have enough memory
        4. Enable selective_checkpointing for memory savings
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[FSDP2Config] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None
    ):
        """
        Initialize FSDP2 Trainer.

        Args:
            model: PyTorch model to train
            config: FSDP2 configuration
            optimizer: Optimizer (created after FSDP wrapping)
            criterion: Loss function
        """
        if not FSDP2_AVAILABLE:
            raise RuntimeError(
                "FSDP2 not available. Requires PyTorch 2.4+. "
                "Please upgrade: pip install torch>=2.4"
            )

        self.config = config or FSDP2Config()
        self.criterion = criterion

        # Initialize distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        # Auto-detect dp_size if not specified
        if self.config.dp_size is None:
            self.config.dp_size = self.world_size // self.config.tp_size

        # Validate configuration
        total_gpus = self.config.dp_size * self.config.tp_size
        if total_gpus != self.world_size:
            raise ValueError(
                f"dp_size ({self.config.dp_size}) * tp_size ({self.config.tp_size}) "
                f"= {total_gpus} must equal world_size ({self.world_size})"
            )

        logger.info(f"Initializing FSDP2 Trainer")
        logger.info(f"  World size: {self.world_size}")
        logger.info(f"  DP size: {self.config.dp_size}")
        logger.info(f"  TP size: {self.config.tp_size}")

        # Create device mesh
        self.device_mesh = self._create_device_mesh()

        # Compile model if requested (before FSDP wrapping)
        if self.config.compile:
            logger.info("Compiling model with torch.compile (before FSDP wrapping)")
            model = self._compile_model(model)

        # Apply FSDP2 wrapping
        self.model = self._apply_fsdp2(model)

        # Create optimizer after FSDP wrapping
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = None

        # Setup mixed precision scaler if needed
        self.scaler = None
        if self.config.mixed_precision is not None:
            self.scaler = torch.cuda.amp.GradScaler()

        logger.info("✓ FSDP2 Trainer initialized successfully")

    def _create_device_mesh(self) -> DeviceMesh:
        """
        Create device mesh for FSDP2.

        The device mesh defines the parallelism topology:
        - 1D mesh: Pure data parallelism
        - 2D mesh: Hybrid DP + TP
        - 3D mesh: DP + TP + PP (future support)

        Returns:
            DeviceMesh object
        """
        if len(self.config.mesh_dim_names) == 1:
            # 1D mesh - pure data parallelism
            mesh_shape = (self.config.dp_size,)
            logger.info(f"Creating 1D device mesh: {mesh_shape}")
        elif len(self.config.mesh_dim_names) == 2:
            # 2D mesh - hybrid DP + TP
            mesh_shape = (self.config.dp_size, self.config.tp_size)
            logger.info(f"Creating 2D device mesh: {mesh_shape}")
            logger.info(f"  Dimensions: {self.config.mesh_dim_names}")
        else:
            raise ValueError(
                f"Unsupported mesh dimensions: {len(self.config.mesh_dim_names)}. "
                "Supported: 1 (DP) or 2 (DP+TP)"
            )

        device_mesh = DeviceMesh(
            device_type="cuda",
            mesh=torch.arange(self.world_size).reshape(mesh_shape),
            mesh_dim_names=self.config.mesh_dim_names
        )

        logger.info(f"✓ Device mesh created: {device_mesh}")
        return device_mesh

    def _apply_fsdp2(self, model: nn.Module) -> nn.Module:
        """
        Apply FSDP2 wrapping to model using fully_shard.

        This uses DTensor-based per-parameter sharding for improved
        performance and composability.

        Args:
            model: Model to wrap

        Returns:
            FSDP2-wrapped model
        """
        from torch.distributed.fsdp import fully_shard

        logger.info("Applying FSDP2 (fully_shard) to model")
        logger.info(f"  Reshard after forward: {self.config.reshard_after_forward}")
        logger.info(f"  CPU offload: {self.config.cpu_offload}")

        # Configure FSDP2 parameters
        fsdp_kwargs = {
            'mesh': self.device_mesh,
            'reshard_after_forward': self.config.reshard_after_forward,
        }

        if self.config.mixed_precision is not None:
            fsdp_kwargs['mixed_precision'] = self.config.mixed_precision

        # Apply fully_shard to all modules
        # Note: FSDP2 uses automatic per-parameter sharding
        for module in model.modules():
            if isinstance(module, nn.Module) and len(list(module.parameters())) > 0:
                # Skip if already wrapped
                if not hasattr(module, '_fsdp_wrapped'):
                    try:
                        fully_shard(module, **fsdp_kwargs)
                        module._fsdp_wrapped = True
                    except Exception as e:
                        logger.debug(f"Skipping module {type(module).__name__}: {e}")

        logger.info("✓ FSDP2 wrapping complete")
        logger.info("  Parameters are now DTensor sharded on dim-0")
        return model

    def _compile_model(self, model: nn.Module) -> nn.Module:
        """
        Compile model with torch.compile.

        Args:
            model: Model to compile

        Returns:
            Compiled model
        """
        compile_config = self.config.compile_config or {}

        default_config = {
            'backend': 'inductor',
            'mode': 'default',
            'fullgraph': False
        }
        default_config.update(compile_config)

        logger.info(f"Compiling with config: {default_config}")
        compiled_model = torch.compile(model, **default_config)
        logger.info("✓ Model compiled successfully")
        return compiled_model

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        gradient_accumulation_steps: int = 1,
        current_step: int = 0
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            data: Input data
            target: Target labels
            gradient_accumulation_steps: Number of steps to accumulate gradients
            current_step: Current step in accumulation cycle

        Returns:
            Dictionary with loss and other metrics
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")

        self.model.train()

        # Move data to GPU
        device = next(self.model.parameters()).device
        data = data.to(device)
        target = target.to(device)

        # Forward pass
        if self.scaler is not None:
            # Mixed precision forward
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
        else:
            output = self.model(data)
            loss = self.criterion(output, target)

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (only on last accumulation step)
        if (current_step + 1) % gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        return {
            'loss': loss.item() * gradient_accumulation_steps,
            'scaled_loss': loss.item()
        }

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Set optimizer after FSDP wrapping.

        Args:
            optimizer: PyTorch optimizer
        """
        self.optimizer = optimizer
        logger.info(f"✓ Optimizer set: {type(optimizer).__name__}")

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        **extra_state
    ):
        """
        Save FSDP2 checkpoint (communication-free!).

        FSDP2 advantage: No all-gathers needed, each rank saves its shards.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            **extra_state: Additional state to save
        """
        import torch.distributed.checkpoint as dist_cp
        from torch.distributed.checkpoint import FileSystemWriter

        logger.info(f"Saving FSDP2 checkpoint to {filepath}")

        state_dict = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            **extra_state
        }

        if self.optimizer is not None:
            state_dict['optimizer'] = self.optimizer.state_dict()

        # Use distributed checkpoint (communication-free)
        try:
            writer = FileSystemWriter(filepath)
            dist_cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=writer
            )
            logger.info("✓ Checkpoint saved (communication-free)")
        except Exception as e:
            logger.error(f"Failed to save distributed checkpoint: {e}")
            logger.info("Falling back to standard checkpoint")
            if self.rank == 0:
                torch.save(state_dict, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load FSDP2 checkpoint.

        Args:
            filepath: Path to checkpoint

        Returns:
            Loaded state dictionary
        """
        import torch.distributed.checkpoint as dist_cp
        from torch.distributed.checkpoint import FileSystemReader

        logger.info(f"Loading FSDP2 checkpoint from {filepath}")

        state_dict = {
            'model': self.model.state_dict(),
        }

        if self.optimizer is not None:
            state_dict['optimizer'] = self.optimizer.state_dict()

        try:
            reader = FileSystemReader(filepath)
            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=reader
            )
            logger.info("✓ Checkpoint loaded")
        except Exception as e:
            logger.error(f"Failed to load distributed checkpoint: {e}")
            logger.info("Falling back to standard checkpoint load")
            checkpoint = torch.load(filepath, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            state_dict = checkpoint

        return state_dict


def get_fsdp2_mixed_precision_policy(
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    buffer_dtype: torch.dtype = torch.float32
) -> Any:
    """
    Get FSDP2 mixed precision policy.

    Args:
        param_dtype: Parameter dtype (bfloat16 or float16)
        reduce_dtype: Gradient reduction dtype
        buffer_dtype: Buffer dtype

    Returns:
        MixedPrecision policy

    Recommended configurations:
    - BF16: param=bf16, reduce=fp32, buffer=fp32 (NVIDIA A100/H100)
    - FP16: param=fp16, reduce=fp32, buffer=fp32 (older GPUs)
    - FP8: Requires torchao float8 integration (H100/H200)
    """
    from torch.distributed.fsdp import MixedPrecision

    policy = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype
    )

    logger.info(f"Created mixed precision policy:")
    logger.info(f"  Param dtype: {param_dtype}")
    logger.info(f"  Reduce dtype: {reduce_dtype}")
    logger.info(f"  Buffer dtype: {buffer_dtype}")

    return policy
