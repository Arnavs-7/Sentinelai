"""Export the best PyTorch RUL model to ONNX and benchmark it.

The best deep model recorded in the production bundle is exported with
dynamic batch and sequence axes, verified for numerical parity against
its PyTorch source with :mod:`onnxruntime`, and benchmarked over 1000
inference runs.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lstm_model import get_lstm_model
from src.models.model_utils import load_model
from src.models.transformer_model import get_transformer_model
from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_CHECKPOINT_DIR = Path("models_saved")
_PRODUCTION_PATH = _CHECKPOINT_DIR / "production_model.pkl"
_EXPORT_DIR = Path("exports")
_ONNX_PATH = _EXPORT_DIR / "sentinel_model.onnx"
_WINDOW_SIZE = 30
_BENCHMARK_RUNS = 1000
_TORCH_BUILDERS = {
    "lstm": get_lstm_model,
    "transformer": get_transformer_model,
}


class PredictionOnly(nn.Module):
    """Wrapper exposing only the scalar RUL output of a model.

    Several SentinelAI models return a ``(prediction, attention)`` tuple;
    this wrapper discards the auxiliary outputs so the graph has a single
    well-defined output for ONNX export.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the wrapper.

        Args:
            model: The wrapped RUL model.
        """
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """Run the wrapped model and return its prediction tensor.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            The prediction tensor of shape ``(batch,)``.
        """
        output = self.model(x)
        if isinstance(output, tuple):
            output = output[0]
        return output


def _select_best_model() -> Tuple[str, int]:
    """Determine which deep model to export and its input size.

    Returns:
        A tuple ``(model_name, input_size)``.

    Raises:
        FileNotFoundError: If no exportable model can be located.
    """
    if not _PRODUCTION_PATH.is_file():
        raise FileNotFoundError(
            f"Production bundle not found: {_PRODUCTION_PATH}. "
            "Run scripts/train_all_models.py first."
        )
    bundle = joblib.load(_PRODUCTION_PATH)
    input_size = int(bundle["input_size"])
    best = str(bundle.get("best_model", "lstm"))

    if best not in _TORCH_BUILDERS:
        for candidate in ("lstm", "transformer"):
            if (_CHECKPOINT_DIR / f"{candidate}_best.pt").is_file():
                best = candidate
                break
        else:
            raise FileNotFoundError("No LSTM or transformer checkpoint found.")
    logger.info("Selected %s for ONNX export (input_size=%d)", best, input_size)
    return best, input_size


def _load_model(name: str, input_size: int) -> nn.Module:
    """Build and load the chosen deep model from its checkpoint.

    Args:
        name: Model name, ``lstm`` or ``transformer``.
        input_size: Number of input features per time step.

    Returns:
        The loaded model wrapped to expose a single output.
    """
    model = _TORCH_BUILDERS[name](input_size, ModelConfig())
    model, _ = load_model(model, _CHECKPOINT_DIR / f"{name}_best.pt")
    model.eval()
    return PredictionOnly(model).eval()


def _export(model: nn.Module, dummy: Tensor) -> None:
    """Export a model to ONNX with dynamic batch and sequence axes.

    Args:
        model: The single-output model to export.
        dummy: A representative example input.
    """
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(_ONNX_PATH),
        input_names=["sensor_window"],
        output_names=["rul"],
        dynamic_axes={
            "sensor_window": {0: "batch", 1: "sequence"},
            "rul": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Exported ONNX model to %s", _ONNX_PATH)


def _verify(
    model: nn.Module, session: ort.InferenceSession, dummy: Tensor
) -> float:
    """Compare PyTorch and ONNX outputs on the same input.

    Args:
        model: The PyTorch model.
        session: The ONNX runtime session.
        dummy: The input tensor used for both models.

    Returns:
        The maximum absolute difference between the two outputs.
    """
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy().reshape(-1)
    onnx_out = session.run(
        None, {"sensor_window": dummy.numpy()}
    )[0].reshape(-1)
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if max_diff < 1e-5:
        logger.info("Verification passed: max_diff=%.3e", max_diff)
    else:
        logger.warning("Verification tolerance exceeded: max_diff=%.3e", max_diff)
    return max_diff


def _benchmark(
    model: nn.Module, session: ort.InferenceSession, dummy: Tensor
) -> Dict[str, float]:
    """Benchmark PyTorch versus ONNX inference latency.

    Args:
        model: The PyTorch model.
        session: The ONNX runtime session.
        dummy: The input used for every inference run.

    Returns:
        A dict with mean latencies, the speedup factor and ``max_diff``.
    """
    inputs = {"sensor_window": dummy.numpy()}

    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        start = time.perf_counter()
        for _ in range(_BENCHMARK_RUNS):
            model(dummy)
        pytorch_ms = (time.perf_counter() - start) / _BENCHMARK_RUNS * 1000.0

    for _ in range(10):
        session.run(None, inputs)
    start = time.perf_counter()
    for _ in range(_BENCHMARK_RUNS):
        session.run(None, inputs)
    onnx_ms = (time.perf_counter() - start) / _BENCHMARK_RUNS * 1000.0

    return {
        "pytorch_ms": round(pytorch_ms, 4),
        "onnx_ms": round(onnx_ms, 4),
        "speedup": round(pytorch_ms / onnx_ms, 3) if onnx_ms else 0.0,
    }


def main() -> None:
    """Export, verify and benchmark the best deep RUL model."""
    torch.manual_seed(_SEED)
    name, input_size = _select_best_model()
    model = _load_model(name, input_size)

    dummy = torch.randn(1, _WINDOW_SIZE, input_size)
    _export(model, dummy)

    session = ort.InferenceSession(
        str(_ONNX_PATH), providers=["CPUExecutionProvider"]
    )
    max_diff = _verify(model, session, dummy)
    results = _benchmark(model, session, dummy)
    results["max_diff"] = max_diff

    logger.info(
        "Benchmark results: pytorch_ms=%.4f onnx_ms=%.4f speedup=%.3f "
        "max_diff=%.3e",
        results["pytorch_ms"],
        results["onnx_ms"],
        results["speedup"],
        results["max_diff"],
    )


if __name__ == "__main__":
    main()
