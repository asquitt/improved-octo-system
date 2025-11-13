"""
Learning Rate Finder

Implements the learning rate range test to find optimal learning rate.
Based on the paper "Cyclical Learning Rates for Training Neural Networks"
and fastai's LR finder.

Author: Your Name
Date: 2025-11
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LRFinder:
    """
    Find optimal learning rate using LR range test.

    The LR range test:
    1. Start with very small LR
    2. Gradually increase LR exponentially
    3. Record loss at each LR
    4. Find LR where loss decreases fastest
    5. Use this LR (or slightly lower) for training

    Example:
    --------
    >>> model = MyModel()
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> lr_finder = LRFinder(model, optimizer, criterion)
    >>> lr_finder.range_test(train_loader)
    >>> lr_finder.plot()
    >>> optimal_lr = lr_finder.get_best_lr()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
    ):
        """
        Initialize LR finder.

        Parameters:
        -----------
        model : nn.Module
            Model to train
        optimizer : torch.optim.Optimizer
            Optimizer
        criterion : nn.Module
            Loss function
        device : str
            Device to use
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.lrs = []
        self.losses = []
        self.best_lr = None

        # Save initial state
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()

    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.05,
    ):
        """
        Perform learning rate range test.

        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        start_lr : float
            Starting learning rate
        end_lr : float
            Ending learning rate
        num_iter : int
            Number of iterations to run
        smooth_f : float
            Smoothing factor for loss (0-1)
        """
        logger.info(f"Running LR range test: {start_lr} to {end_lr}")

        self.model.to(self.device)
        self.model.train()

        # Calculate LR multiplier
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)

        # Set initial LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = start_lr

        current_lr = start_lr
        best_loss = float('inf')
        smoothed_loss = 0

        iterator = iter(train_loader)

        for iteration in range(num_iter):
            # Get batch
            try:
                data, target = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                data, target = next(iterator)

            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Compute smoothed loss
            if iteration == 0:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = smooth_f * loss.item() + (1 - smooth_f) * smoothed_loss

            # Stop if loss explodes
            if smoothed_loss > 4 * best_loss or torch.isnan(loss):
                logger.info("Loss exploded, stopping early")
                break

            # Record
            self.lrs.append(current_lr)
            self.losses.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Increase LR
            current_lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            if (iteration + 1) % 10 == 0:
                logger.info(f"Iter {iteration + 1}/{num_iter} | LR: {current_lr:.2e} | Loss: {smoothed_loss:.4f}")

        # Restore original state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        logger.info("LR range test complete")

    def plot(self, save_path: str = "lr_finder.png", skip_start: int = 10, skip_end: int = 5):
        """
        Plot learning rate vs loss.

        Parameters:
        -----------
        save_path : str
            Path to save plot
        skip_start : int
            Skip first N points (unstable)
        skip_end : int
            Skip last N points (exploding)
        """
        if not self.lrs:
            logger.warning("No LR range test data. Run range_test() first.")
            return

        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(lrs, losses, linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Rate Finder', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark suggested LR
        best_lr = self.get_best_lr()
        if best_lr:
            ax.axvline(x=best_lr, color='r', linestyle='--', label=f'Suggested LR: {best_lr:.2e}')
            ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved LR finder plot to {save_path}")

        return fig, ax

    def get_best_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """
        Get the best learning rate (steepest gradient).

        Parameters:
        -----------
        skip_start : int
            Skip first N points
        skip_end : int
            Skip last N points

        Returns:
        --------
        float : Suggested learning rate
        """
        if not self.lrs:
            return None

        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]

        # Find steepest gradient
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)

        best_lr = lrs[min_gradient_idx]
        self.best_lr = best_lr

        logger.info(f"Suggested learning rate: {best_lr:.2e}")

        return best_lr
