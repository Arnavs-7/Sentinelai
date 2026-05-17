"""Shared utilities for model seeding, device handling and persistence."""

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import torch
from torch import nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42


def set_seed(seed: int = _SEED) -> None:
    """Seed the Python, NumPy and PyTorch random number generators.

    Args:
        seed: Seed value applied to every generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Set global random seed to %d", seed)


def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Preference order is CUDA, then Apple MPS, then CPU.

    Returns:
        The selected :class:`torch.device`.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and (
        torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Selected compute device: %s", device)
    return device


def count_parameters(model: nn.Module) -> int:
    """Count the trainable parameters of a model.

    Args:
        model: The model to inspect.

    Returns:
        The total number of parameters with ``requires_grad`` set.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, path: Path, metadata: Dict[str, Any]) -> None:
    """Persist a model's state dict alongside arbitrary metadata.

    Args:
        model: The model whose ``state_dict`` is saved.
        path: Destination checkpoint file.
        metadata: Arbitrary serializable metadata stored with the weights.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "metadata": metadata}
    torch.save(checkpoint, path)
    logger.info("Saved model checkpoint to %s", path)


def load_model(model: nn.Module, path: Path) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load weights into a model and return it with its metadata.

    Args:
        model: The model instance to populate with loaded weights.
        path: Source checkpoint file produced by :func:`save_model`.

    Returns:
        A tuple ``(model, metadata)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    metadata: Dict[str, Any] = checkpoint.get("metadata", {})
    logger.info("Loaded model checkpoint from %s", path)
    return model, metadata


def save_sklearn_model(model: Any, path: Path) -> None:
    """Persist a scikit-learn-style model with joblib.

    Args:
        model: The estimator to serialize.
        path: Destination file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved sklearn model to %s", path)


def load_sklearn_model(path: Path) -> Any:
    """Load a joblib-serialized scikit-learn-style model.

    Args:
        path: Source file produced by :func:`save_sklearn_model`.

    Returns:
        The deserialized estimator.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Sklearn model file not found: {path}")
    model = joblib.load(path)
    logger.info("Loaded sklearn model from %s", path)
    return model
