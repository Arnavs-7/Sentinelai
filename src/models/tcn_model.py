"""Temporal Convolutional Network for RUL regression."""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import weight_norm

from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_DILATIONS: List[int] = [1, 2, 4, 8, 16, 32]
_MC_PASSES = 10


class CausalConv1d(nn.Module):
    """A 1-D convolution with strictly causal left-only padding.

    Padding of ``(kernel_size - 1) * dilation`` is applied to the left so
    that each output step depends only on current and past inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        """Initialize the causal convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel width.
            dilation: Dilation factor of the kernel.
        """
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the causal convolution.

        Args:
            x: Input tensor of shape ``(batch, in_channels, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, seq_len)``.
        """
        padded = F.pad(x, (self.left_padding, 0))
        return self.conv(padded)


class TCNResidualBlock(nn.Module):
    """A residual block of two weight-normalized causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel width.
            dilation: Dilation factor of both convolutions.
            dropout: Dropout probability applied after each activation.
        """
        super().__init__()
        self.conv1 = weight_norm(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation).conv
        )
        self.pad1 = (kernel_size - 1) * dilation
        self.conv2 = weight_norm(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation).conv
        )
        self.pad2 = (kernel_size - 1) * dilation
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape ``(batch, in_channels, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, seq_len)``.
        """
        out = self.conv1(F.pad(x, (self.pad1, 0)))
        out = self.dropout1(self.relu1(out))
        out = self.conv2(F.pad(out, (self.pad2, 0)))
        out = self.dropout2(self.relu2(out))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


class TemporalConvNet(nn.Module):
    """Stacked dilated causal convolutions with an MLP regression head."""

    def __init__(self, input_size: int, config: ModelConfig) -> None:
        """Initialize the network.

        Args:
            input_size: Number of input features per time step.
            config: Shared model configuration.
        """
        super().__init__()
        self.config = config
        kernel_size = config.tcn_kernel_size
        channels = config.tcn_channels

        blocks: List[nn.Module] = []
        prev_channels = input_size
        for dilation, out_channels in zip(_DILATIONS, channels):
            blocks.append(
                TCNResidualBlock(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            prev_channels = out_channels
        self.network = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels[-1], 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

        receptive_field = (
            sum((kernel_size - 1) * d * 2 for d in _DILATIONS) + 1
        )
        logger.info("TemporalConvNet receptive field: %d", receptive_field)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Predicted RUL tensor of shape ``(batch,)``.
        """
        x = x.permute(0, 2, 1)
        features = self.network(x)
        pooled = self.pool(features).flatten(start_dim=1)
        hidden = self.relu(self.fc1(pooled))
        return self.fc_out(hidden).squeeze(-1)

    def enable_mc_dropout(self) -> None:
        """Set every dropout layer to train mode for MC-dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict(self, x: Tensor) -> Dict[str, np.ndarray]:
        """Predict RUL with dropout-based uncertainty.

        Dropout is left active and 10 stochastic forward passes are
        aggregated into a mean estimate with 5%/95% bounds.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A dict with ``rul``, ``lower`` and ``upper`` arrays.
        """
        self.eval()
        self.enable_mc_dropout()

        predictions = []
        for _ in range(_MC_PASSES):
            predictions.append(self.forward(x).cpu().numpy())

        stacked = np.stack(predictions, axis=0)
        logger.info("Ran %d dropout passes for prediction", _MC_PASSES)
        return {
            "rul": stacked.mean(axis=0),
            "lower": np.percentile(stacked, 5, axis=0),
            "upper": np.percentile(stacked, 95, axis=0),
        }


def get_tcn_model(input_size: int, config: ModelConfig) -> TemporalConvNet:
    """Construct a :class:`TemporalConvNet` model.

    Args:
        input_size: Number of input features per time step.
        config: Shared model configuration.

    Returns:
        An initialized :class:`TemporalConvNet` instance.
    """
    torch.manual_seed(_SEED)
    model = TemporalConvNet(input_size=input_size, config=config)
    logger.info("Built TemporalConvNet with input_size=%d", input_size)
    return model
