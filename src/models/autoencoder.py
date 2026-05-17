"""LSTM and variational LSTM autoencoders for sensor anomaly detection."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder with a reconstruction threshold.

    The encoder compresses a window to a fixed latent vector; the decoder
    reconstructs the window from it. Reconstruction error gives a per-sample
    anomaly score compared against a calibrated threshold.
    """

    def __init__(self, input_size: int, config: ModelConfig) -> None:
        """Initialize the autoencoder.

        Args:
            input_size: Number of input features per time step.
            config: Shared model configuration.
        """
        super().__init__()
        self.config = config
        self.input_size = input_size
        hidden, latent = config.ae_encoder_hidden

        self.encoder_lstm1 = nn.LSTM(
            input_size, hidden, batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(hidden, latent, batch_first=True)
        self.decoder_lstm1 = nn.LSTM(latent, hidden, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(hidden, input_size, batch_first=True)

        self.hidden_size = hidden
        self.latent_size = latent
        self.threshold: float = 0.0

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode and reconstruct an input window.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A tuple ``(reconstructed, latent)`` with shapes
            ``(batch, seq_len, input_size)`` and ``(batch, latent_size)``.
        """
        seq_len = x.size(1)
        enc, _ = self.encoder_lstm1(x)
        _, (hidden, _) = self.encoder_lstm2(enc)
        latent = hidden[-1]

        repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)
        dec, _ = self.decoder_lstm1(repeated)
        reconstructed, _ = self.decoder_lstm2(dec)
        return reconstructed, latent

    def compute_anomaly_score(self, x: Tensor) -> Tensor:
        """Compute a per-sample reconstruction anomaly score.

        The squared error is averaged over features at each time step and
        then over time, yielding one scalar score per window.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A tensor of shape ``(batch,)`` of anomaly scores.
        """
        reconstructed, _ = self.forward(x)
        per_timestep = ((x - reconstructed) ** 2).mean(dim=2)
        return per_timestep.mean(dim=1)

    def set_threshold(
        self, val_scores: np.ndarray, percentile: float = 95.0
    ) -> None:
        """Calibrate the anomaly threshold from validation scores.

        Args:
            val_scores: Anomaly scores from nominal validation data.
            percentile: Percentile of ``val_scores`` used as the threshold.
        """
        self.threshold = float(np.percentile(val_scores, percentile))
        logger.info(
            "Set anomaly threshold to %.6f (p%.1f)",
            self.threshold,
            percentile,
        )

    @torch.no_grad()
    def is_anomaly(self, x: Tensor) -> Tuple[bool, float]:
        """Classify a single window as anomalous or nominal.

        Args:
            x: Input tensor of shape ``(1, seq_len, input_size)`` or
                ``(seq_len, input_size)``.

        Returns:
            A tuple ``(is_anomaly, score)``.
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        score = float(self.compute_anomaly_score(x).mean().item())
        return score > self.threshold, score


class VariationalLSTMAutoencoder(nn.Module):
    """Variational LSTM autoencoder with a probabilistic latent space.

    The encoder produces a Gaussian latent distribution; samples are decoded
    back to the input window. The evidence lower bound combines a
    reconstruction term with a KL-divergence regularizer.
    """

    def __init__(self, input_size: int, config: ModelConfig) -> None:
        """Initialize the variational autoencoder.

        Args:
            input_size: Number of input features per time step.
            config: Shared model configuration.
        """
        super().__init__()
        self.config = config
        self.input_size = input_size
        hidden, latent = config.ae_encoder_hidden

        self.encoder_lstm1 = nn.LSTM(
            input_size, hidden, batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_log_var = nn.Linear(hidden, latent)

        self.decoder_lstm1 = nn.LSTM(latent, hidden, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(hidden, input_size, batch_first=True)

        self.hidden_size = hidden
        self.latent_size = latent

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode an input window into latent distribution parameters.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A tuple ``(mu, log_var)`` each of shape ``(batch, latent_size)``.
        """
        enc, _ = self.encoder_lstm1(x)
        _, (hidden, _) = self.encoder_lstm2(enc)
        last = hidden[-1]
        return self.fc_mu(last), self.fc_log_var(last)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample a latent vector using the reparameterization trick.

        Args:
            mu: Latent mean tensor of shape ``(batch, latent_size)``.
            log_var: Latent log-variance tensor of the same shape.

        Returns:
            A sampled latent tensor of shape ``(batch, latent_size)``.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, seq_len: int) -> Tensor:
        """Decode a latent vector into a reconstructed window.

        Args:
            z: Latent tensor of shape ``(batch, latent_size)``.
            seq_len: Length of the sequence to reconstruct.

        Returns:
            A reconstructed tensor of shape ``(batch, seq_len, input_size)``.
        """
        repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        dec, _ = self.decoder_lstm1(repeated)
        reconstructed, _ = self.decoder_lstm2(dec)
        return reconstructed

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode, sample and reconstruct an input window.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            A tuple ``(reconstructed, mu, log_var)``.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z, x.size(1))
        return reconstructed, mu, log_var

    def loss(
        self,
        x: Tensor,
        reconstructed: Tensor,
        mu: Tensor,
        log_var: Tensor,
        beta: float = 1.0,
    ) -> Tensor:
        """Compute the beta-weighted variational autoencoder loss.

        Args:
            x: Original input tensor.
            reconstructed: Reconstructed tensor from :meth:`forward`.
            mu: Latent mean tensor.
            log_var: Latent log-variance tensor.
            beta: Weight applied to the KL-divergence term.

        Returns:
            A scalar loss tensor.
        """
        reconstruction = F.mse_loss(reconstructed, x, reduction="mean")
        kl = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        return reconstruction + beta * kl

    @torch.no_grad()
    def anomaly_score(self, x: Tensor, n_samples: int = 10) -> Tensor:
        """Estimate a per-sample anomaly score by latent sampling.

        Reconstruction error is averaged over ``n_samples`` draws from the
        latent distribution to marginalize sampling noise.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.
            n_samples: Number of latent samples to average over.

        Returns:
            A tensor of shape ``(batch,)`` of anomaly scores.
        """
        self.eval()
        mu, log_var = self.encode(x)
        scores = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, log_var)
            reconstructed = self.decode(z, x.size(1))
            per_timestep = ((x - reconstructed) ** 2).mean(dim=2)
            scores.append(per_timestep.mean(dim=1))
        return torch.stack(scores, dim=0).mean(dim=0)


def get_autoencoder(
    input_size: int, config: ModelConfig
) -> LSTMAutoencoder:
    """Construct an :class:`LSTMAutoencoder` model.

    Args:
        input_size: Number of input features per time step.
        config: Shared model configuration.

    Returns:
        An initialized :class:`LSTMAutoencoder` instance.
    """
    torch.manual_seed(_SEED)
    model = LSTMAutoencoder(input_size=input_size, config=config)
    logger.info("Built LSTMAutoencoder with input_size=%d", input_size)
    return model


def get_vae(
    input_size: int, config: ModelConfig
) -> VariationalLSTMAutoencoder:
    """Construct a :class:`VariationalLSTMAutoencoder` model.

    Args:
        input_size: Number of input features per time step.
        config: Shared model configuration.

    Returns:
        An initialized :class:`VariationalLSTMAutoencoder` instance.
    """
    torch.manual_seed(_SEED)
    model = VariationalLSTMAutoencoder(input_size=input_size, config=config)
    logger.info(
        "Built VariationalLSTMAutoencoder with input_size=%d", input_size
    )
    return model
