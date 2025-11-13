"""
Training Visualization Tools

Create beautiful visualizations for training metrics, GPU utilization,
model architecture, and performance comparisons.

Features:
---------
- Training/validation curves
- GPU utilization plots
- Throughput comparisons
- Loss landscapes
- Learning rate schedules
- Multi-run comparisons

Author: Your Name
Date: 2025-11
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """
    Visualize training metrics and performance.

    This class provides tools to create publication-quality plots
    for training analysis and comparison.

    Example:
    --------
    >>> viz = TrainingVisualizer()
    >>> viz.add_run("DDP", losses=[0.5, 0.3, 0.2], accuracies=[0.8, 0.9, 0.95])
    >>> viz.add_run("FSDP", losses=[0.5, 0.3, 0.19], accuracies=[0.8, 0.91, 0.96])
    >>> viz.plot_comparison()
    >>> viz.save("comparison.png")
    """

    def __init__(self):
        """Initialize visualizer."""
        self.runs = {}
        self.fig = None
        self.axes = None

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')

        logger.info("TrainingVisualizer initialized")

    def add_run(
        self,
        name: str,
        losses: List[float],
        accuracies: Optional[List[float]] = None,
        val_losses: Optional[List[float]] = None,
        val_accuracies: Optional[List[float]] = None,
        learning_rates: Optional[List[float]] = None,
    ):
        """
        Add a training run to visualize.

        Parameters:
        -----------
        name : str
            Run identifier
        losses : List[float]
            Training losses per epoch
        accuracies : List[float], optional
            Training accuracies per epoch
        val_losses : List[float], optional
            Validation losses
        val_accuracies : List[float], optional
            Validation accuracies
        learning_rates : List[float], optional
            Learning rates per epoch
        """
        self.runs[name] = {
            "losses": losses,
            "accuracies": accuracies or [],
            "val_losses": val_losses or [],
            "val_accuracies": val_accuracies or [],
            "learning_rates": learning_rates or [],
        }

        logger.info(f"Added run: {name}")

    def plot_comparison(
        self,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
    ):
        """
        Plot comparison of all runs.

        Creates a 2x2 grid showing:
        - Training loss
        - Validation loss
        - Training accuracy
        - Validation accuracy

        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot training losses
        for name, metrics in self.runs.items():
            epochs = range(1, len(metrics["losses"]) + 1)
            axes[0, 0].plot(epochs, metrics["losses"], label=name, linewidth=2, marker='o')

        axes[0, 0].set_xlabel("Epoch", fontsize=12)
        axes[0, 0].set_ylabel("Training Loss", fontsize=12)
        axes[0, 0].set_title("Training Loss Comparison", fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot validation losses
        for name, metrics in self.runs.items():
            if metrics["val_losses"]:
                epochs = range(1, len(metrics["val_losses"]) + 1)
                axes[0, 1].plot(epochs, metrics["val_losses"], label=name, linewidth=2, marker='s')

        axes[0, 1].set_xlabel("Epoch", fontsize=12)
        axes[0, 1].set_ylabel("Validation Loss", fontsize=12)
        axes[0, 1].set_title("Validation Loss Comparison", fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot training accuracies
        for name, metrics in self.runs.items():
            if metrics["accuracies"]:
                epochs = range(1, len(metrics["accuracies"]) + 1)
                axes[1, 0].plot(epochs, metrics["accuracies"], label=name, linewidth=2, marker='^')

        axes[1, 0].set_xlabel("Epoch", fontsize=12)
        axes[1, 0].set_ylabel("Training Accuracy", fontsize=12)
        axes[1, 0].set_title("Training Accuracy Comparison", fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot validation accuracies
        for name, metrics in self.runs.items():
            if metrics["val_accuracies"]:
                epochs = range(1, len(metrics["val_accuracies"]) + 1)
                axes[1, 1].plot(epochs, metrics["val_accuracies"], label=name, linewidth=2, marker='D')

        axes[1, 1].set_xlabel("Epoch", fontsize=12)
        axes[1, 1].set_ylabel("Validation Accuracy", fontsize=12)
        axes[1, 1].set_title("Validation Accuracy Comparison", fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")

        self.fig = fig
        self.axes = axes

        return fig, axes

    def plot_throughput_comparison(
        self,
        strategies: List[str],
        throughputs: List[float],
        save_path: Optional[str] = None,
    ):
        """
        Plot throughput comparison bar chart.

        Parameters:
        -----------
        strategies : List[str]
            Strategy names
        throughputs : List[float]
            Samples per second for each strategy
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
        bars = ax.bar(strategies, throughputs, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel("Throughput (samples/sec)", fontsize=14, fontweight='bold')
        ax.set_title("Training Throughput Comparison", fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved throughput plot to {save_path}")

        return fig, ax

    def save(self, filepath: str):
        """Save current figure."""
        if self.fig:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {filepath}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: str = "training_curves.png",
):
    """
    Plot training curves for a single run.

    Parameters:
    -----------
    train_losses : List[float]
        Training losses
    val_losses : List[float], optional
        Validation losses
    train_accs : List[float], optional
        Training accuracies
    val_accs : List[float], optional
        Validation accuracies
    save_path : str
        Save path
    """
    num_plots = 1
    if val_losses:
        num_plots += 1
    if train_accs:
        num_plots += 1
    if val_accs:
        num_plots += 1

    fig, axes = plt.subplots(1, min(num_plots, 2), figsize=(14, 5))

    if num_plots == 1:
        axes = [axes]

    # Training loss
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')

    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        axes[0].plot(val_epochs, val_losses, 'r--', label='Val Loss', linewidth=2, marker='s')

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracies (if provided)
    if train_accs and len(axes) > 1:
        epochs = range(1, len(train_accs) + 1)
        axes[1].plot(epochs, train_accs, 'g-', label='Train Acc', linewidth=2, marker='^')

        if val_accs:
            val_epochs = range(1, len(val_accs) + 1)
            axes[1].plot(val_epochs, val_accs, 'm--', label='Val Acc', linewidth=2, marker='D')

        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Training Accuracy", fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training curves to {save_path}")

    return fig, axes


def plot_gpu_utilization(
    gpu_utils: List[float],
    timestamps: Optional[List[float]] = None,
    save_path: str = "gpu_utilization.png",
):
    """
    Plot GPU utilization over time.

    Parameters:
    -----------
    gpu_utils : List[float]
        GPU utilization percentages
    timestamps : List[float], optional
        Timestamps (seconds)
    save_path : str
        Save path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if timestamps is None:
        timestamps = list(range(len(gpu_utils)))

    ax.plot(timestamps, gpu_utils, 'b-', linewidth=2)
    ax.fill_between(timestamps, gpu_utils, alpha=0.3)

    ax.axhline(y=80, color='g', linestyle='--', label='Target (80%)', linewidth=2)
    ax.axhline(y=50, color='orange', linestyle='--', label='Inefficient (<50%)', linewidth=2)

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("GPU Utilization (%)", fontsize=12)
    ax.set_title("GPU Utilization Over Time", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved GPU utilization plot to {save_path}")

    return fig, ax


def create_performance_dashboard(
    metrics: Dict[str, any],
    save_path: str = "performance_dashboard.png",
):
    """
    Create a comprehensive performance dashboard.

    Parameters:
    -----------
    metrics : dict
        Dictionary containing all metrics
    save_path : str
        Save path

    Example metrics dict:
    ---------------------
    {
        "train_losses": [0.5, 0.3, 0.2],
        "val_losses": [0.6, 0.4, 0.25],
        "gpu_util": [75, 80, 82],
        "throughput": [100, 120, 125],
        "memory_usage": [8, 8.5, 9],
    }
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Training/Validation Loss
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(metrics["train_losses"]) + 1)
    ax1.plot(epochs, metrics["train_losses"], 'b-', label='Train', linewidth=2, marker='o')
    if "val_losses" in metrics:
        ax1.plot(epochs, metrics["val_losses"], 'r--', label='Val', linewidth=2, marker='s')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progress", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # GPU Utilization
    ax2 = fig.add_subplot(gs[0, 2])
    if "gpu_util" in metrics:
        ax2.bar(["GPU"], [np.mean(metrics["gpu_util"])], color='green', alpha=0.7)
        ax2.set_ylabel("Utilization (%)")
        ax2.set_title("Avg GPU Util", fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.5)

    # Throughput
    ax3 = fig.add_subplot(gs[1, :])
    if "throughput" in metrics:
        iterations = range(len(metrics["throughput"]))
        ax3.plot(iterations, metrics["throughput"], 'g-', linewidth=2)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Samples/sec")
        ax3.set_title("Training Throughput", fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # Memory Usage
    ax4 = fig.add_subplot(gs[2, :])
    if "memory_usage" in metrics:
        iterations = range(len(metrics["memory_usage"]))
        ax4.plot(iterations, metrics["memory_usage"], 'm-', linewidth=2)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Memory (GB)")
        ax4.set_title("GPU Memory Usage", fontweight='bold')
        ax4.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved performance dashboard to {save_path}")

    return fig
