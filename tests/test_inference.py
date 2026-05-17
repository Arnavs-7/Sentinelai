"""Tests for inference: prediction, explainability and drift detection."""

from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from src.inference.drift_detector import DriftDetector
from src.inference.explainer import SentinelExplainer
from src.inference.predictor import Predictor
from src.utils.config import InferenceConfig


class _ConstantModel:
    """A callable model that always returns a fixed RUL tensor."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a constant prediction tensor.

        Args:
            tensor: The (ignored) input tensor.

        Returns:
            A single-element prediction tensor.
        """
        return torch.tensor([92.0])


def _build_predictor() -> Predictor:
    """Build a predictor with a stubbed model and preprocessing step.

    Returns:
        A :class:`Predictor` ready for ``predict_single`` calls.
    """
    predictor = Predictor(InferenceConfig())
    predictor._model_cache = {"primary": _ConstantModel(), "_metadata": {}}
    predictor.preprocess = lambda data: torch.zeros(1, 30, 5)  # type: ignore
    return predictor


def test_predictor_output_format() -> None:
    """``predict_single`` returns every required result key."""
    predictor = _build_predictor()
    result = predictor.predict_single("machine-1", [[0.0] * 14])
    required = {
        "machine_id",
        "predicted_rul",
        "confidence_interval",
        "risk_level",
        "risk_score",
        "model_contributions",
        "anomaly_score",
        "timestamp",
    }
    assert required.issubset(result.keys())
    assert isinstance(result["predicted_rul"], float)


def test_risk_levels() -> None:
    """Risk levels map correctly across the RUL range."""
    predictor = Predictor(InferenceConfig())
    assert predictor._get_risk_level(150.0)[0] == "HEALTHY"
    assert predictor._get_risk_level(50.0)[0] == "WARNING"
    assert predictor._get_risk_level(10.0)[0] == "CRITICAL"


def test_shap_explainer() -> None:
    """The SHAP explainer runs without error on a dummy tree model."""
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(40, 5))
    y = X.sum(axis=1) + rng.normal(0.0, 0.1, size=40)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    feature_names = [f"feature_{i}" for i in range(5)]
    explainer = SentinelExplainer(models={"rf": model}, feature_names=feature_names)
    result = explainer.shap_explain(model, X, "rf")
    assert "shap_values" in result
    assert result["model_type"] == "rf"


def test_drift_detector_no_drift() -> None:
    """Identical reference and live data raise no drift flag."""
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 1.0, size=(300, 3))
    detector = DriftDetector(reference, ["a", "b", "c"])
    report = detector.full_report(reference, [1.0] * 8)
    assert report["summary"]["drift_detected"] is False


def test_drift_detector_drift() -> None:
    """A strongly shifted live distribution raises a drift flag."""
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 1.0, size=(300, 3))
    live = reference + 8.0
    detector = DriftDetector(reference, ["a", "b", "c"])
    report = detector.full_report(live, [0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0])
    assert report["summary"]["drift_detected"] is True


def test_natural_language_output() -> None:
    """The natural-language explanation is a non-empty string."""
    explainer = SentinelExplainer(models={}, feature_names=[])
    top_features: List[Dict[str, Any]] = [
        {"name": "sensor_4", "contribution": 0.31, "direction": "up", "value": 0.18},
        {"name": "sensor_9", "contribution": 0.22, "direction": "down", "value": -0.11},
    ]
    text = explainer.natural_language_explanation(top_features, 95.0)
    assert isinstance(text, str)
    assert len(text) > 0
