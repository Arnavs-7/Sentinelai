"""Training callbacks: early stopping, checkpointing and LR scheduling."""

import math
from pathlib import Path
from typing import List

from torch import nn
from torch.optim import Optimizer

from src.models.model_utils import save_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Improvement is measured against the best score seen so far; once
    ``patience`` epochs pass without an improvement exceeding
    ``min_delta`` the callback signals that training should stop.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        """Initialize the callback.

        Args:
            patience: Number of stale epochs tolerated before stopping.
            min_delta: Minimum change counted as an improvement.
            mode: ``min`` to minimize the metric, ``max`` to maximize it.

        Raises:
            ValueError: If ``mode`` is not ``min`` or ``max``.
        """
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float = math.inf if mode == "min" else -math.inf
        self.counter: int = 0
        self.best_epoch: int = 0
        self._epoch: int = 0

    def _is_improvement(self, metric: float) -> bool:
        """Return whether a metric value improves on the best score.

        Args:
            metric: The latest monitored metric value.

        Returns:
            ``True`` if the value is a sufficient improvement.
        """
        if self.mode == "min":
            return metric < self.best_score - self.min_delta
        return metric > self.best_score + self.min_delta

    def __call__(self, metric: float) -> bool:
        """Update internal state with a new metric and test for stopping.

        Args:
            metric: The latest monitored metric value.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        self._epoch += 1
        if self._is_improvement(metric):
            self.best_score = metric
            self.best_epoch = self._epoch
            self.counter = 0
            return False

        self.counter += 1
        logger.info(
            "EarlyStopping: no improvement for %d/%d epochs",
            self.counter,
            self.patience,
        )
        if self.counter >= self.patience:
            logger.info(
                "EarlyStopping triggered; best epoch was %d (%.6f)",
                self.best_epoch,
                self.best_score,
            )
            return True
        return False


class ModelCheckpoint:
    """Persist a model's weights whenever the monitored metric improves."""

    def __init__(
        self,
        save_path: Path,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        """Initialize the callback.

        Args:
            save_path: Destination checkpoint file.
            monitor: Name of the metric being monitored, stored in
                checkpoint metadata.
            mode: ``min`` to save on decreases, ``max`` on increases.

        Raises:
            ValueError: If ``mode`` is not ``min`` or ``max``.
        """
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.save_path = Path(save_path)
        self.monitor = monitor
        self.mode = mode
        self.best_score: float = math.inf if mode == "min" else -math.inf
        self.best_epoch: int = 0

    def _is_improvement(self, metric: float) -> bool:
        """Return whether a metric value improves on the best score.

        Args:
            metric: The latest monitored metric value.

        Returns:
            ``True`` if the value beats the stored best score.
        """
        if self.mode == "min":
            return metric < self.best_score
        return metric > self.best_score

    def __call__(
        self, metric: float, model: nn.Module, epoch: int
    ) -> None:
        """Save the model if the monitored metric improved.

        Args:
            metric: The latest monitored metric value.
            model: The model to checkpoint.
            epoch: The current epoch index.
        """
        if not self._is_improvement(metric):
            return
        self.best_score = metric
        self.best_epoch = epoch
        save_model(
            model,
            self.save_path,
            metadata={
                "monitor": self.monitor,
                "metric": metric,
                "epoch": epoch,
            },
        )
        logger.info(
            "ModelCheckpoint: saved epoch %d with %s=%.6f",
            epoch,
            self.monitor,
            metric,
        )


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay learning-rate scheduler.

    For the first ``warmup_steps`` the learning rate rises linearly from
    zero to the optimizer's base rate; afterwards it follows a cosine
    curve down to ``min_lr`` at ``total_steps``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: The optimizer whose learning rate is scheduled.
            warmup_steps: Number of linear warmup steps.
            total_steps: Total number of training steps.
            min_lr: Floor learning rate reached at the end of decay.
        """
        self.optimizer = optimizer
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, self.warmup_steps + 1)
        self.min_lr = min_lr
        self.base_lrs: List[float] = [
            group["lr"] for group in optimizer.param_groups
        ]
        self._step_count: int = 0
        self._last_lr: float = 0.0
        self._apply_lr()

    def _compute_scale(self) -> float:
        """Compute the multiplicative scale for the current step.

        Returns:
            A factor in ``[0, 1]`` applied to each base learning rate.
        """
        step = self._step_count
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply_lr(self) -> None:
        """Write the scheduled learning rate into the optimizer."""
        scale = self._compute_scale()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = self.min_lr + (base_lr - self.min_lr) * scale
            group["lr"] = lr
            self._last_lr = lr

    def step(self) -> None:
        """Advance the scheduler by one step and update the optimizer."""
        self._step_count += 1
        self._apply_lr()

    def get_lr(self) -> float:
        """Return the most recently applied learning rate.

        Returns:
            The current learning rate of the first parameter group.
        """
        return self._last_lr
