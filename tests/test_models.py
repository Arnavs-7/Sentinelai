"""Tests for the SentinelAI model architectures and persistence."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import Tensor, nn

from src.models.autoencoder import get_autoencoder
from src.models.lstm_model import get_lstm_model
from src.models.model_utils import load_model, save_model
from src.models.tcn_model import get_tcn_model
from src.models.transformer_model import get_transformer_model
from src.utils.config import ModelConfig

_BATCH = 4
_SEQ = 30
_FEATURES = 50


@pytest.fixture
def config() -> ModelConfig:
    """Return a default model configuration.

    Returns:
        A :class:`ModelConfig` instance.
    """
    return ModelConfig()


@pytest.fixture
def dummy_input() -> Tensor:
    """Return a dummy input tensor of shape (4, 30, 50).

    Returns:
        A float tensor of random values.
    """
    torch.manual_seed(42)
    return torch.randn(_BATCH, _SEQ, _FEATURES)


def test_lstm_forward(config: ModelConfig, dummy_input: Tensor) -> None:
    """The LSTM produces one scalar prediction per batch element."""
    model = get_lstm_model(_FEATURES, config).eval()
    prediction, attention = model(dummy_input)
    assert prediction.reshape(-1).shape == (_BATCH,)
    assert attention.shape == (_BATCH, _SEQ)


def test_tcn_forward(config: ModelConfig, dummy_input: Tensor) -> None:
    """The TCN produces one scalar prediction per batch element."""
    model = get_tcn_model(_FEATURES, config).eval()
    prediction = model(dummy_input)
    assert prediction.reshape(-1).shape == (_BATCH,)


def test_transformer_forward(
    config: ModelConfig, dummy_input: Tensor
) -> None:
    """The transformer produces one scalar prediction per batch element."""
    model = get_transformer_model(_FEATURES, config).eval()
    prediction, _ = model(dummy_input)
    assert prediction.reshape(-1).shape == (_BATCH,)


def test_autoencoder_forward(
    config: ModelConfig, dummy_input: Tensor
) -> None:
    """The autoencoder reconstructs an output matching the input shape."""
    model = get_autoencoder(_FEATURES, config).eval()
    reconstructed, latent = model(dummy_input)
    assert reconstructed.shape == dummy_input.shape
    assert latent.shape[0] == _BATCH


def test_lstm_mc_dropout(config: ModelConfig, dummy_input: Tensor) -> None:
    """Monte Carlo dropout yields finite mean and standard deviation."""
    model = get_lstm_model(_FEATURES, config)
    result = model.predict(dummy_input)
    assert np.isfinite(result["rul"]).all()
    assert np.isfinite(result["std"]).all()
    assert result["rul"].shape == (_BATCH,)


def test_model_save_load(
    config: ModelConfig, dummy_input: Tensor, tmp_path: Path
) -> None:
    """A saved model reloads to identical outputs within tolerance."""
    source = get_lstm_model(_FEATURES, config).eval()
    with torch.no_grad():
        expected, _ = source(dummy_input)

    checkpoint = tmp_path / "lstm.pt"
    save_model(source, checkpoint, metadata={"epoch": 1})

    restored = get_lstm_model(_FEATURES, config)
    restored, metadata = load_model(restored, checkpoint)
    restored.eval()
    with torch.no_grad():
        actual, _ = restored(dummy_input)

    assert metadata["epoch"] == 1
    assert torch.allclose(expected, actual, atol=1e-4)


def test_onnx_export(
    config: ModelConfig, dummy_input: Tensor, tmp_path: Path
) -> None:
    """An exported LSTM matches its ONNX runtime outputs within 1e-5."""

    class _PredictionOnly(nn.Module):
        """Wrapper exposing only the model's prediction output."""

        def __init__(self, model: nn.Module) -> None:
            """Initialize the wrapper.

            Args:
                model: The wrapped model.
            """
            super().__init__()
            self.model = model

        def forward(self, x: Tensor) -> Tensor:
            """Return only the prediction tensor.

            Args:
                x: Input tensor.

            Returns:
                The prediction tensor.
            """
            output = self.model(x)
            return output[0] if isinstance(output, tuple) else output

    model = _PredictionOnly(get_lstm_model(_FEATURES, config)).eval()
    onnx_path = tmp_path / "lstm.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["sensor_window"],
        output_names=["rul"],
        dynamic_axes={
            "sensor_window": {0: "batch", 1: "sequence"},
            "rul": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy().reshape(-1)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_out = session.run(
        None, {"sensor_window": dummy_input.numpy()}
    )[0].reshape(-1)

    assert np.max(np.abs(torch_out - onnx_out)) < 1e-5
