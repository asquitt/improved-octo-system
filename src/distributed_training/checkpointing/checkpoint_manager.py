"""
Checkpoint Management for Distributed Training

Handles saving, loading, and managing checkpoints with:
- Automatic checkpoint rotation (keep only N recent)
- Distributed checkpoint sharding
- Failure recovery
- Best model tracking

Author: Your Name
Date: 2025-11
"""

import os
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for distributed training.

    Features:
    - Automatic saving at intervals
    - Keep only N most recent checkpoints
    - Track best model by metric
    - Distributed-safe (only rank 0 saves)
    - Supports checkpoint sharding for large models

    Example:
    --------
    >>> manager = CheckpointManager(
    ...     checkpoint_dir="./checkpoints",
    ...     max_checkpoints=3
    ... )
    >>> manager.save_checkpoint(
    ...     epoch=5,
    ...     model=model,
    ...     optimizer=optimizer,
    ...     metrics={"loss": 0.5, "accuracy": 0.95}
    ... )
    >>> state = manager.load_latest_checkpoint()
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 3,
        best_metric: str = "loss",
        best_mode: str = "min",
    ):
        """
        Initialize checkpoint manager.

        Parameters:
        -----------
        checkpoint_dir : str
            Directory to save checkpoints
        max_checkpoints : int
            Maximum number of checkpoints to keep (0 = keep all)
        best_metric : str
            Metric to track for best model
        best_mode : str
            'min' or 'max' for best metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.best_mode = best_mode

        # Get distributed info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Track best model
        self.best_metric_value = float("inf") if best_mode == "min" else float("-inf")
        self.best_checkpoint_path = None

        # Create checkpoint directory
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Save a checkpoint.

        Only the main process (rank 0) saves to avoid conflicts.

        Parameters:
        -----------
        epoch : int
            Current epoch
        model : torch.nn.Module
            Model to save
        optimizer : torch.optim.Optimizer
            Optimizer to save
        metrics : dict, optional
            Training metrics (loss, accuracy, etc.)
        extra_state : dict, optional
            Additional state to save

        Returns:
        --------
        str : Path to saved checkpoint (None for non-main processes)
        """
        if not self.is_main_process:
            return None

        # Unwrap DDP model if necessary
        if hasattr(model, "module"):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
            "extra_state": extra_state or {},
        }

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save metadata
        self._save_metadata(epoch, metrics)

        # Check if this is the best model
        if metrics and self.best_metric in metrics:
            metric_value = metrics[self.best_metric]
            if self._is_better(metric_value, self.best_metric_value):
                self.best_metric_value = metric_value
                self.best_checkpoint_path = checkpoint_path
                # Save copy as best model
                best_path = self.checkpoint_dir / "best_model.pt"
                shutil.copy(checkpoint_path, best_path)
                logger.info(
                    f"New best model! {self.best_metric}={metric_value:.4f}"
                )

        # Clean up old checkpoints
        if self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Parameters:
        -----------
        checkpoint_path : str
            Path to checkpoint file
        model : torch.nn.Module
            Model to load state into
        optimizer : torch.optim.Optimizer, optional
            Optimizer to load state into

        Returns:
        --------
        dict : Checkpoint metadata (epoch, metrics, etc.)
        """
        logger.info(f"[Rank {self.rank}] Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu",
        )

        # Load model state
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(
            f"[Rank {self.rank}] Loaded checkpoint from epoch {checkpoint['epoch']}"
        )

        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint.get("metrics", {}),
            "extra_state": checkpoint.get("extra_state", {}),
        }

    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Returns:
        --------
        dict or None : Checkpoint metadata, or None if no checkpoints found
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        if not checkpoints:
            logger.warning("No checkpoints found")
            return None

        latest_checkpoint = checkpoints[-1]
        return self.load_checkpoint(str(latest_checkpoint), model, optimizer)

    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.

        Returns:
        --------
        dict or None : Checkpoint metadata, or None if no best checkpoint
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            logger.warning("No best checkpoint found")
            return None

        return self.load_checkpoint(str(best_path), model, optimizer)

    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.best_mode == "min":
            return current < best
        else:
            return current > best

    def _save_metadata(self, epoch: int, metrics: Optional[Dict[str, float]]):
        """Save checkpoint metadata."""
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"

        # Load existing metadata
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"checkpoints": []}

        # Add new checkpoint
        metadata["checkpoints"].append(
            {"epoch": epoch, "metrics": metrics or {}}
        )

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        # Keep only recent checkpoints
        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[: -self.max_checkpoints]:
                # Don't delete best checkpoint
                if checkpoint != Path(self.best_checkpoint_path):
                    checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint}")


def save_sharded_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    epoch: int,
):
    """
    Save checkpoint with model sharding for very large models.

    Each GPU saves its own shard of the model parameters.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to save
    optimizer : torch.optim.Optimizer
        Optimizer to save
    checkpoint_dir : str
        Directory to save checkpoints
    epoch : int
        Current epoch
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save this rank's shard
    shard_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_rank_{rank}.pt"

    checkpoint = {
        "epoch": epoch,
        "rank": rank,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, shard_path)
    logger.info(f"[Rank {rank}] Saved checkpoint shard: {shard_path}")
