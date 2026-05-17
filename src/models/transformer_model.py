"""Transformer encoder with sinusoidal positional encoding for RUL regression."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_MC_PASSES = 10


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings.

    Even dimensions use a sine wave and odd dimensions a cosine wave, each
    with a geometrically decreasing frequency, so absolute and relative
    positions are encoded without any learned parameters.
    """

    def __init__(
        self, d_model: int, max_len: int = 5000, dropout: float = 0.1
    ) -> None:
        """Initialize the positional encoding.

        Args:
            d_model: Embedding dimensionality.
            max_len: Maximum sequence length supported.
            dropout: Dropout probability applied to the encoded tensor.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to the input embeddings.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            The positionally encoded tensor with dropout applied.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder over sensor windows with an MLP regression head.

    Inputs are projected to ``d_model``, positionally encoded, processed by a
    stack of transformer encoder layers, mean-pooled over time and decoded to
    a scalar RUL estimate.
    """

    def __init__(self, input_size: int, config: ModelConfig) -> None:
        """Initialize the transformer.

        Args:
            input_size: Number of input features per time step.
            config: Shared model configuration.
        """
        super().__init__()
        self.config = config
        d_model = config.transformer_d_model
        nhead = config.transformer_nhead
        num_layers = config.transformer_num_layers
        dropout = config.transformer_dropout

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc1 = nn.Linear(d_model, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.
            return_attention: If ``True``, also return the per-layer
                self-attention weight maps captured via forward hooks.

        Returns:
            A tuple ``(prediction, attention_maps)`` where ``prediction`` has
            shape ``(batch,)`` and ``attention_maps`` is a list of attention
            tensors or ``None`` when ``return_attention`` is ``False``.
        """
        hidden = self.input_projection(x)
        hidden = self.pos_encoding(hidden)

        attention_maps: Optional[List[Tensor]] = None
        handles: List[torch.utils.hooks.RemovableHandle] = []
        if return_attention:
            attention_maps = []

            def make_hook(store: List[Tensor]):
                """Build a forward hook that records attention weights.

                Args:
                    store: List that captured attention tensors append to.

                Returns:
                    The hook callable bound to ``store``.
                """

                def hook(
                    module: nn.Module,
                    inputs: Tuple[Tensor, ...],
                    output: Tuple[Tensor, ...],
                ) -> None:
                    """Record the attention weights from a layer's output."""
                    if isinstance(output, tuple) and len(output) > 1:
                        weights = output[1]
                        if weights is not None:
                            store.append(weights.detach())

                return hook

            for layer in self.encoder.layers:
                handles.append(
                    layer.self_attn.register_forward_hook(
                        make_hook(attention_maps)
                    )
                )

        encoded = self.encoder(hidden)

        for handle in handles:
            handle.remove()

        pooled = encoded.mean(dim=1)
        head = self.relu(self.fc1(pooled))
        prediction = self.fc_out(head).squeeze(-1)
        return prediction, attention_maps

    def enable_mc_dropout(self) -> None:
        """Set every dropout layer to train mode for MC-dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict(self, x: Tensor) -> Dict[str, object]:
        """Predict RUL with dropout-based uncertainty.

        Dropout is left active and 10 stochastic forward passes are
        aggregated into a mean estimate with 5%/95% bounds; attention maps
        are captured from a final deterministic-style pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A dict with ``rul``, ``lower``, ``upper`` arrays and the
            ``attention_maps`` list from the last pass.
        """
        self.eval()
        self.enable_mc_dropout()

        predictions: List[np.ndarray] = []
        attention_maps: Optional[List[Tensor]] = None
        for index in range(_MC_PASSES):
            collect = index == _MC_PASSES - 1
            pred, attn = self.forward(x, return_attention=collect)
            predictions.append(pred.cpu().numpy())
            if collect and attn is not None:
                attention_maps = [a.cpu() for a in attn]

        stacked = np.stack(predictions, axis=0)
        logger.info("Ran %d dropout passes for prediction", _MC_PASSES)
        return {
            "rul": stacked.mean(axis=0),
            "lower": np.percentile(stacked, 5, axis=0),
            "upper": np.percentile(stacked, 95, axis=0),
            "attention_maps": attention_maps if attention_maps else [],
        }


def get_transformer_model(
    input_size: int, config: ModelConfig
) -> TimeSeriesTransformer:
    """Construct a :class:`TimeSeriesTransformer` model.

    Args:
        input_size: Number of input features per time step.
        config: Shared model configuration.

    Returns:
        An initialized :class:`TimeSeriesTransformer` instance.
    """
    torch.manual_seed(_SEED)
    model = TimeSeriesTransformer(input_size=input_size, config=config)
    logger.info("Built TimeSeriesTransformer with input_size=%d", input_size)
    return model
