"""Evaluate every saved SentinelAI model and generate a report bundle.

The script rebuilds the test split, loads each persisted model, scores
them with :class:`ModelEvaluator`, writes Markdown, HTML and JSON
reports plus per-model diagnostic plots, and logs a summary table.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_all_models import _build_datasets, _metrics, _torch_predict

from src.models.baseline import RandomForestRUL, XGBoostRUL
from src.models.lstm_model import get_lstm_model
from src.models.model_utils import load_model
from src.models.tcn_model import get_tcn_model
from src.models.transformer_model import get_transformer_model
from src.training.evaluator import ModelEvaluator
from src.utils.config import DataConfig, ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CHECKPOINT_DIR = Path("models_saved")
_PRODUCTION_PATH = _CHECKPOINT_DIR / "production_model.pkl"
_REPORTS_DIR = Path("reports")
_PLOTS_DIR = _REPORTS_DIR / "plots"
_TORCH_BUILDERS = {
    "lstm": get_lstm_model,
    "tcn": get_tcn_model,
    "transformer": get_transformer_model,
}


def _load_models(input_size: int) -> Dict[str, object]:
    """Load every persisted model that can be found on disk.

    Args:
        input_size: Number of input features per time step.

    Returns:
        A mapping of model name to the loaded model instance.
    """
    config = ModelConfig()
    models: Dict[str, object] = {}

    for name, builder in _TORCH_BUILDERS.items():
        path = _CHECKPOINT_DIR / f"{name}_best.pt"
        if not path.is_file():
            logger.warning("Skipping %s: checkpoint %s not found", name, path)
            continue
        model = builder(input_size, config)
        model, _ = load_model(model, path)
        model.eval()
        models[name] = model
        logger.info("Loaded torch model %s", name)

    for name, cls in (("xgboost", XGBoostRUL), ("rf", RandomForestRUL)):
        suffix = "joblib"
        path = _CHECKPOINT_DIR / f"{name}.{suffix}"
        if not path.is_file():
            logger.warning("Skipping %s: file %s not found", name, path)
            continue
        model = cls(config)
        model.load(path)
        models[name] = model
        logger.info("Loaded tree model %s", name)

    return models


def _plot_predicted_vs_actual(
    name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """Save a predicted-versus-actual scatter plot for one model.

    Args:
        name: Model name used in the title and filename.
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4, color="#2c7fb8")
    limit = float(max(y_true.max(), y_pred.max(), 1.0))
    ax.plot([0, limit], [0, limit], color="#e74c3c", linestyle="--")
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"Predicted vs Actual — {name}")
    fig.tight_layout()
    fig.savefig(_PLOTS_DIR / f"predicted_vs_actual_{name}.png", dpi=120)
    plt.close(fig)


def _plot_error_distribution(name: str, errors: np.ndarray) -> None:
    """Save an error-distribution histogram for one model.

    Args:
        name: Model name used in the title and filename.
        errors: Prediction errors (predicted minus actual).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=40, color="#7fcdbb", edgecolor="#2c7fb8")
    ax.axvline(0.0, color="#e74c3c", linestyle="--")
    ax.set_xlabel("Prediction error (cycles)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution — {name}")
    fig.tight_layout()
    fig.savefig(_PLOTS_DIR / f"error_distribution_{name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Run evaluation for every saved model and write the report bundle."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if _PRODUCTION_PATH.is_file():
        bundle = joblib.load(_PRODUCTION_PATH)
        input_size = int(bundle.get("input_size", 0))
    else:
        bundle = {}
        input_size = 0

    loaders, tensors, feature_cols = _build_datasets()
    if not input_size:
        input_size = len(feature_cols)

    models = _load_models(input_size)
    if not models:
        logger.error("No saved models found in %s", _CHECKPOINT_DIR)
        return

    evaluator = ModelEvaluator(DataConfig())
    X_test, y_test_tensor = tensors["test"]
    y_test = y_test_tensor.numpy()
    X_test_2d = X_test[:, -1, :].numpy()

    results: Dict[str, Dict[str, object]] = {}
    for name, model in models.items():
        if isinstance(model, torch.nn.Module):
            y_pred = _torch_predict(model, X_test)
        else:
            y_pred = np.asarray(model.predict(X_test_2d)).reshape(-1)

        metrics = _metrics(y_test, y_pred)
        metrics["model"] = name
        metrics["calibration_error"] = ModelEvaluator._calibration_error(
            y_test, y_pred
        )
        results[name] = metrics

        errors = y_pred - y_test
        _plot_predicted_vs_actual(name, y_test, y_pred)
        _plot_error_distribution(name, errors)
        logger.info(
            "Evaluated %s: RMSE=%.4f MAE=%.4f score_s=%.1f",
            name,
            metrics["rmse"],
            metrics["mae"],
            metrics["score_s"],
        )

    evaluator.save_report(results, _REPORTS_DIR / "evaluation_report.md")
    (_REPORTS_DIR / "evaluation_report.json").write_text(
        json.dumps(list(results.values()), indent=2, default=str),
        encoding="utf-8",
    )

    summary = pd.DataFrame(list(results.values())).sort_values("rmse")
    logger.info(
        "=== Evaluation summary ===\n%s", summary.to_string(index=False)
    )
    logger.info(
        "Reports written to %s (markdown, html, json) with plots in %s",
        _REPORTS_DIR,
        _PLOTS_DIR,
    )


if __name__ == "__main__":
    main()
