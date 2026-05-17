"""Loss functions for RUL regression training."""

from typing import List

import torch
from torch import Tensor, nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RMSELoss(nn.Module):
    """Root mean squared error loss."""

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the loss.

        Args:
            eps: Small constant added before the square root for numerical
                stability.
        """
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the RMSE between predictions and targets.

        Args:
            pred: Predicted RUL tensor.
            target: Ground-truth RUL tensor.

        Returns:
            A scalar RMSE loss tensor.
        """
        return torch.sqrt(self.mse(pred, target) + self.eps)


class AsymmetricLoss(nn.Module):
    """Asymmetric squared loss penalizing early predictions more heavily.

    In predictive maintenance an early prediction (under-estimating RUL)
    is conservative but costly, while a late prediction risks failure.
    Here early predictions are scaled by ``late_penalty`` and late
    predictions by ``1.0``.
    """

    def __init__(self, late_penalty: float = 1.5) -> None:
        """Initialize the loss.

        Args:
            late_penalty: Multiplier applied to early-prediction errors.
        """
        super().__init__()
        self.late_penalty = late_penalty

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the asymmetric loss.

        Args:
            pred: Predicted RUL tensor.
            target: Ground-truth RUL tensor.

        Returns:
            A scalar asymmetric loss tensor.
        """
        error = pred - target
        squared = error ** 2
        weight = torch.where(
            pred < target,
            torch.full_like(squared, self.late_penalty),
            torch.ones_like(squared),
        )
        return (weight * squared).mean()


class HuberLoss(nn.Module):
    """Huber loss combining squared and absolute error regimes."""

    def __init__(self, delta: float = 1.0) -> None:
        """Initialize the loss.

        Args:
            delta: Threshold at which the loss transitions from quadratic
                to linear.
        """
        super().__init__()
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the Huber loss.

        Args:
            pred: Predicted RUL tensor.
            target: Ground-truth RUL tensor.

        Returns:
            A scalar Huber loss tensor.
        """
        error = torch.abs(pred - target)
        quadratic = torch.minimum(error, torch.tensor(self.delta))
        linear = error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


class QuantileLoss(nn.Module):
    """Pinball loss for multi-quantile regression."""

    def __init__(
        self, quantiles: List[float] = [0.05, 0.5, 0.95]
    ) -> None:
        """Initialize the loss.

        Args:
            quantiles: Quantile levels predicted by the model, aligned to
                the last dimension of the prediction tensor.
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the averaged pinball loss.

        Args:
            pred: Predicted tensor of shape ``(batch, n_quantiles)``.
            target: Ground-truth RUL tensor of shape ``(batch,)``.

        Returns:
            A scalar quantile loss tensor.
        """
        target = target.unsqueeze(-1)
        losses = []
        for index, q in enumerate(self.quantiles):
            error = target - pred[:, index : index + 1]
            losses.append(torch.maximum(q * error, (q - 1.0) * error))
        return torch.cat(losses, dim=1).mean()


class CombinedLoss(nn.Module):
    """Convex combination of asymmetric and quantile-median losses."""

    def __init__(self, alpha: float = 0.7) -> None:
        """Initialize the loss.

        Args:
            alpha: Weight on the asymmetric term; ``1 - alpha`` weights the
                quantile median term.
        """
        super().__init__()
        self.alpha = alpha
        self.asymmetric = AsymmetricLoss()
        self.quantile = QuantileLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the combined loss.

        The point prediction is taken as the median quantile column when
        ``pred`` is multi-quantile, otherwise ``pred`` itself.

        Args:
            pred: Predicted tensor, either ``(batch,)`` or
                ``(batch, n_quantiles)``.
            target: Ground-truth RUL tensor of shape ``(batch,)``.

        Returns:
            A scalar combined loss tensor.
        """
        if pred.dim() == 2 and pred.size(1) == len(self.quantile.quantiles):
            median_index = self.quantile.quantiles.index(0.5)
            point = pred[:, median_index]
            quantile_term = self.quantile(pred, target)
        else:
            point = pred.reshape(-1)
            quantile_term = self.quantile(
                point.unsqueeze(-1).repeat(1, len(self.quantile.quantiles)),
                target,
            )
        asymmetric_term = self.asymmetric(point, target)
        return self.alpha * asymmetric_term + (1.0 - self.alpha) * quantile_term
