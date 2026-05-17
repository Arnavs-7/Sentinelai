"""Bidirectional LSTM with Bahdanau attention for RUL regression."""

from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_MC_PASSES = 50


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention over an encoder sequence.

    A learned scoring function combines a single query vector with every
    key vector; the softmax-normalized scores weight the keys into a
    context vector.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize the attention layer.

        Args:
            hidden_size: Dimensionality of the query and key vectors.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query: Tensor, keys: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the context vector and attention weights.

        Args:
            query: Query tensor of shape ``(batch, hidden_size)``.
            keys: Key tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            A tuple ``(context, attention_weights)`` with shapes
            ``(batch, hidden_size)`` and ``(batch, seq_len)``.
        """
        query_expanded = query.unsqueeze(1)
        score = torch.tanh(self.W1(query_expanded) + self.W2(keys))
        weights = torch.softmax(self.V(score).squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class LSTMWithAttention(nn.Module):
    """Bidirectional LSTM encoder with attention and an MLP regression head.

    The final attention context is decoded by a three-layer MLP into a
    scalar RUL estimate; Monte Carlo dropout provides predictive
    uncertainty at inference time.
    """

    def __init__(self, input_size: int, config: ModelConfig) -> None:
        """Initialize the model.

        Args:
            input_size: Number of input features per time step.
            config: Shared model configuration.
        """
        super().__init__()
        self.config = config
        hidden_size = config.lstm_hidden_size
        self.encoder_dim = hidden_size * 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
        )
        self.attention = BahdanauAttention(self.encoder_dim)
        self.dropout = nn.Dropout(config.lstm_dropout)
        self.fc1 = nn.Linear(self.encoder_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A tuple ``(prediction, attention_weights)`` with shapes
            ``(batch,)`` and ``(batch, seq_len)``.
        """
        encoder_out, _ = self.lstm(x)
        query = encoder_out[:, -1, :]
        context, attention_weights = self.attention(query, encoder_out)

        hidden = self.dropout(context)
        hidden = self.relu(self.fc1(hidden))
        hidden = self.relu(self.fc2(hidden))
        prediction = self.fc_out(hidden).squeeze(-1)
        return prediction, attention_weights

    def enable_mc_dropout(self) -> None:
        """Set every dropout layer to train mode for MC-dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict(self, x: Tensor) -> Dict[str, np.ndarray]:
        """Predict RUL with Monte Carlo dropout uncertainty.

        Dropout is left active and 50 stochastic forward passes are
        aggregated into a mean, standard deviation and 5%/95% bounds.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A dict with ``rul``, ``std``, ``lower``, ``upper`` arrays and
            the mean ``attention`` weights.
        """
        self.eval()
        self.enable_mc_dropout()

        predictions = []
        attentions = []
        for _ in range(_MC_PASSES):
            pred, attn = self.forward(x)
            predictions.append(pred.cpu().numpy())
            attentions.append(attn.cpu().numpy())

        stacked = np.stack(predictions, axis=0)
        attention = np.stack(attentions, axis=0).mean(axis=0)
        logger.info("Ran %d MC-dropout passes for prediction", _MC_PASSES)
        return {
            "rul": stacked.mean(axis=0),
            "std": stacked.std(axis=0),
            "lower": np.percentile(stacked, 5, axis=0),
            "upper": np.percentile(stacked, 95, axis=0),
            "attention": attention,
        }


def get_lstm_model(input_size: int, config: ModelConfig) -> LSTMWithAttention:
    """Construct an :class:`LSTMWithAttention` model.

    Args:
        input_size: Number of input features per time step.
        config: Shared model configuration.

    Returns:
        An initialized :class:`LSTMWithAttention` instance.
    """
    torch.manual_seed(_SEED)
    model = LSTMWithAttention(input_size=input_size, config=config)
    logger.info("Built LSTMWithAttention with input_size=%d", input_size)
    return model
