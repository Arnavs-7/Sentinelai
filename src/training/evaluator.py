"""Evaluation metrics and reporting for RUL models."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.config import DataConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_EARLY_FACTOR = 13.0
_LATE_FACTOR = 10.0


class ModelEvaluator:
    """Compute RUL metrics and assemble comparison and per-engine reports.

    Reported metrics cover point accuracy (RMSE, MAE, R^2), the NASA
    asymmetric scoring function, tolerance accuracies and a calibration
    error derived from the empirical residual distribution.
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialize the evaluator.

        Args:
            config: Data configuration providing the RUL clipping range.
        """
        self.config = config

    @staticmethod
    def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the NASA C-MAPSS asymmetric scoring function.

        Late predictions (over-estimating RUL) are penalized more steeply
        than early predictions.

        Args:
            y_true: Ground-truth RUL values.
            y_pred: Predicted RUL values.

        Returns:
            The summed asymmetric score; lower is better.
        """
        diff = y_pred - y_true
        score = np.where(
            diff < 0,
            np.exp(-diff / _EARLY_FACTOR) - 1.0,
            np.exp(diff / _LATE_FACTOR) - 1.0,
        )
        return float(np.sum(score))

    @staticmethod
    def _accuracy_within(
        y_true: np.ndarray, y_pred: np.ndarray, tolerance: float
    ) -> float:
        """Compute the fraction of predictions within an absolute tolerance.

        Args:
            y_true: Ground-truth RUL values.
            y_pred: Predicted RUL values.
            tolerance: Absolute error tolerance in cycles.

        Returns:
            The fraction of samples whose absolute error is within
            ``tolerance``.
        """
        if y_true.size == 0:
            return 0.0
        return float(np.mean(np.abs(y_pred - y_true) <= tolerance))

    @staticmethod
    def _calibration_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Estimate calibration error from standardized residuals.

        Residuals are standardized and compared against the standard
        normal CDF across deciles; the mean absolute deviation between
        expected and observed coverage is returned.

        Args:
            y_true: Ground-truth RUL values.
            y_pred: Predicted RUL values.

        Returns:
            The mean absolute calibration deviation in ``[0, 1]``.
        """
        residuals = y_pred - y_true
        std = residuals.std()
        if std == 0 or residuals.size == 0:
            return 0.0
        standardized = (residuals - residuals.mean()) / std
        levels = np.linspace(0.1, 0.9, 9)
        deviations = []
        for level in levels:
            threshold = np.quantile(standardized, level)
            observed = float(np.mean(standardized <= threshold))
            deviations.append(abs(observed - level))
        return float(np.mean(deviations))

    def _metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute the full metric suite for a set of predictions.

        Args:
            y_true: Ground-truth RUL values.
            y_pred: Predicted RUL values.

        Returns:
            A dict of metric name to value.
        """
        errors = y_pred - y_true
        rmse = float(np.sqrt(np.mean(errors ** 2))) if errors.size else 0.0
        mae = float(np.mean(np.abs(errors))) if errors.size else 0.0
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) if y_true.size else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "score_s": self.nasa_score(y_true, y_pred),
            "acc_10": self._accuracy_within(y_true, y_pred, 10.0),
            "acc_15": self._accuracy_within(y_true, y_pred, 15.0),
            "calibration_error": self._calibration_error(y_true, y_pred),
        }

    @staticmethod
    def _predict_loader(
        model: nn.Module, test_loader: DataLoader
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Run a model over a loader and collect predictions and targets.

        Args:
            model: The model to evaluate.
            test_loader: Loader over the evaluation split.

        Returns:
            A tuple ``(y_true, y_pred)`` of concatenated arrays.
        """
        model.eval()
        preds: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                if isinstance(output, tuple):
                    output = output[0]
                preds.append(output.cpu().numpy().reshape(-1))
                targets.append(batch_y.cpu().numpy().reshape(-1))
        y_pred = np.concatenate(preds) if preds else np.array([])
        y_true = np.concatenate(targets) if targets else np.array([])
        return y_true, y_pred

    def evaluate(
        self, model: nn.Module, test_loader: DataLoader, model_name: str
    ) -> Dict[str, Any]:
        """Evaluate a single model on a test loader.

        Args:
            model: The model to evaluate.
            test_loader: Loader over the test split.
            model_name: Human-readable name recorded in the result.

        Returns:
            A dict of metrics augmented with the ``model`` name.
        """
        y_true, y_pred = self._predict_loader(model, test_loader)
        metrics = self._metrics(y_true, y_pred)
        metrics["model"] = model_name
        logger.info(
            "Evaluated %s: RMSE=%.4f MAE=%.4f score_s=%.1f",
            model_name,
            metrics["rmse"],
            metrics["mae"],
            metrics["score_s"],
        )
        return metrics

    def evaluate_all_models(
        self, models: Dict[str, nn.Module], test_loader: DataLoader
    ) -> pd.DataFrame:
        """Evaluate several models and assemble a comparison table.

        Args:
            models: Mapping of model name to model instance.
            test_loader: Loader over the test split shared by all models.

        Returns:
            A frame with one row per model, sorted by ascending RMSE.
        """
        rows = [
            self.evaluate(model, test_loader, name)
            for name, model in models.items()
        ]
        frame = pd.DataFrame(rows)
        ordered_columns = [
            "model",
            "rmse",
            "mae",
            "r2",
            "score_s",
            "acc_10",
            "acc_15",
            "calibration_error",
        ]
        frame = frame[ordered_columns]
        return frame.sort_values("rmse").reset_index(drop=True)

    def per_engine_report(
        self, model: nn.Module, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Break down prediction error per engine unit.

        Args:
            model: The model to evaluate; called on a feature tensor built
                from the windowed columns of ``test_df``.
            test_df: Frame with an ``engine_id`` column, an ``rul`` target
                column and a ``window`` column of per-sample feature
                arrays.

        Returns:
            A frame with per-engine RMSE, MAE, NASA score and sample
            counts, sorted by descending RMSE.
        """
        windows = np.stack(test_df["window"].to_list(), axis=0)
        targets = test_df["rul"].to_numpy(dtype=float)
        model.eval()
        with torch.no_grad():
            output = model(torch.as_tensor(windows, dtype=torch.float32))
            if isinstance(output, tuple):
                output = output[0]
        predictions = output.cpu().numpy().reshape(-1)

        report = pd.DataFrame(
            {
                "engine_id": test_df["engine_id"].to_numpy(),
                "y_true": targets,
                "y_pred": predictions,
            }
        )
        rows = []
        for engine_id, group in report.groupby("engine_id"):
            y_true = group["y_true"].to_numpy()
            y_pred = group["y_pred"].to_numpy()
            errors = y_pred - y_true
            rows.append(
                {
                    "engine_id": engine_id,
                    "n_samples": len(group),
                    "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "mae": float(np.mean(np.abs(errors))),
                    "score_s": self.nasa_score(y_true, y_pred),
                }
            )
        frame = pd.DataFrame(rows)
        return frame.sort_values("rmse", ascending=False).reset_index(
            drop=True
        )

    def save_report(self, results: Dict[str, Any], path: Path) -> None:
        """Write an evaluation report as paired Markdown and HTML files.

        Args:
            results: Mapping of model name to its metric dict.
            path: Destination Markdown file; the HTML report is written
                alongside it with an ``.html`` suffix.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(list(results.values()))
        markdown = "# Model Evaluation Report\n\n"
        markdown += frame.to_markdown(index=False)
        markdown += "\n"
        path.write_text(markdown, encoding="utf-8")

        html = "<html><head><title>Model Evaluation Report</title></head>"
        html += "<body><h1>Model Evaluation Report</h1>"
        html += frame.to_html(index=False)
        html += "</body></html>"
        html_path = path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")

        logger.info("Saved evaluation report to %s and %s", path, html_path)
