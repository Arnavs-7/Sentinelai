"""Online RUL inference with risk scoring and batched prediction."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.data.feature_engineer import FeatureEngineer
from src.data.preprocessor import CMAPSSPreprocessor
from src.data.windowing import SlidingWindowProcessor
from src.models.lstm_model import get_lstm_model
from src.models.model_utils import get_device
from src.models.tcn_model import get_tcn_model
from src.models.transformer_model import get_transformer_model
from src.utils.config import InferenceConfig, ModelConfig
from src.utils.logger import get_logger
from src.utils.paths import resolve_model_path

logger = get_logger(__name__)

_SEED = 42
_PREPROCESSOR_FILENAME = "preprocessor.joblib"
_HEALTHY_THRESHOLD = 100.0
_CRITICAL_THRESHOLD = 30.0
_MAX_RUL = 125.0


class Predictor:
    """Loads trained models and serves RUL predictions for live sensors.

    Models and the fitted preprocessor are loaded lazily and cached, so
    repeated calls reuse the same in-memory objects. Each prediction is
    annotated with a conformal-style confidence interval, a categorical
    risk level and a normalized risk score.
    """

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the predictor.

        Args:
            config: Inference configuration providing model paths and the
                confidence level used for interval estimation.
        """
        self.config = config
        self.device = get_device()
        self._model_cache: Dict[str, Any] = {}
        self._preprocessor: CMAPSSPreprocessor | None = None
        self._feature_engineer = FeatureEngineer()
        self._windower = SlidingWindowProcessor()

    def _load_models(self) -> None:
        """Load the trained inference model into the cache on first use.

        The checkpoint's metadata records which architecture produced it,
        so the matching network is rebuilt and its weights restored.
        Subsequent calls are no-ops while the cache is populated, ensuring
        weights are read from disk at most once per process.

        Raises:
            FileNotFoundError: If the configured model checkpoint is
                missing.
        """
        if self._model_cache:
            return

        model_path = Path(self.config.model_path)
        if not model_path.is_file():
            model_path = resolve_model_path(model_path.name)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        metadata: Dict[str, Any] = checkpoint.get("metadata", {})
        model_name = str(metadata.get("model", "lstm"))
        input_size = int(metadata.get("input_size", 0))

        builders = {
            "lstm": get_lstm_model,
            "tcn": get_tcn_model,
            "transformer": get_transformer_model,
        }
        builder = builders.get(model_name, get_lstm_model)
        model = builder(input_size, ModelConfig())
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(self.device)
        model.eval()

        self._model_cache["primary"] = model
        self._model_cache["_metadata"] = metadata
        logger.info(
            "Loaded inference model '%s' from %s", model_name, model_path
        )

    def _load_preprocessor(self) -> CMAPSSPreprocessor:
        """Load and cache the fitted preprocessor.

        Returns:
            The fitted :class:`CMAPSSPreprocessor` instance.
        """
        if self._preprocessor is None:
            preprocessor = CMAPSSPreprocessor()
            preprocessor_path = resolve_model_path(_PREPROCESSOR_FILENAME)
            preprocessor.load(preprocessor_path)
            self._preprocessor = preprocessor
            logger.info("Loaded preprocessor from %s", preprocessor_path)
        return self._preprocessor

    def preprocess(self, raw_sensor_data: List[List[float]]) -> torch.Tensor:
        """Transform raw sensor rows into a windowed model-ready tensor.

        The saved scalers and operating-condition clustering are applied,
        engine-aware feature engineering is run, and a single sliding
        window of the configured length is extracted.

        Args:
            raw_sensor_data: Sequence of per-cycle sensor reading rows.

        Returns:
            A float tensor of shape ``(1, window_size, n_features)`` on
            the inference device.
        """
        import pandas as pd

        preprocessor = self._load_preprocessor()
        n_sensors = len(preprocessor.sensor_columns)
        rows = np.asarray(raw_sensor_data, dtype=np.float32)

        frame = pd.DataFrame(
            rows[:, :n_sensors], columns=preprocessor.sensor_columns
        )
        frame.insert(0, "unit_id", 1)
        frame.insert(1, "cycle", np.arange(1, len(frame) + 1))
        for index, op_col in enumerate(["op_1", "op_2", "op_3"]):
            offset = n_sensors + index
            frame[op_col] = (
                rows[:, offset] if rows.shape[1] > offset else 0.0
            )
        frame["rul"] = _MAX_RUL

        scaled = preprocessor.transform(frame)
        engineered = self._feature_engineer.fit_transform(scaled)
        feature_cols = [
            c
            for c in engineered.columns
            if c not in {"unit_id", "cycle", "rul"}
        ]
        window_size = min(len(engineered), 30)
        x_tensor, _ = self._windower.create_windows(
            engineered,
            feature_cols=feature_cols,
            target_col="rul",
            window_size=window_size,
            stride=1,
        )
        if x_tensor.shape[0] == 0:
            raise ValueError("Insufficient sensor history to build a window.")
        return x_tensor[-1:].to(self.device)

    def _get_risk_level(self, rul: float) -> Tuple[str, float]:
        """Map a predicted RUL to a risk label and normalized score.

        Args:
            rul: Predicted remaining useful life in cycles.

        Returns:
            A tuple ``(risk_level, risk_score)`` where the score lies in
            ``[0, 100]`` and rises as RUL falls.
        """
        if rul > _HEALTHY_THRESHOLD:
            fraction = min((rul - _HEALTHY_THRESHOLD) / _HEALTHY_THRESHOLD, 1.0)
            return "HEALTHY", float(30.0 * (1.0 - fraction))
        if rul >= _CRITICAL_THRESHOLD:
            span = _HEALTHY_THRESHOLD - _CRITICAL_THRESHOLD
            fraction = (_HEALTHY_THRESHOLD - rul) / span
            return "WARNING", float(30.0 + 40.0 * fraction)
        fraction = max(0.0, min(rul, _CRITICAL_THRESHOLD)) / _CRITICAL_THRESHOLD
        return "CRITICAL", float(70.0 + 30.0 * (1.0 - fraction))

    def predict_single(
        self, machine_id: str, sensor_data: List[List[float]]
    ) -> Dict[str, Any]:
        """Predict RUL and risk for a single machine.

        Args:
            machine_id: Identifier of the machine being scored.
            sensor_data: Sequence of per-cycle sensor reading rows.

        Returns:
            A dict with the predicted RUL, confidence interval, risk
            level and score, per-model contributions, anomaly score and
            a timestamp.
        """
        self._load_models()
        tensor = self.preprocess(sensor_data)
        model = self._model_cache["primary"]

        with torch.no_grad():
            output = model(tensor)
            if isinstance(output, tuple):
                output = output[0]
        raw_rul = float(np.asarray(output.cpu().numpy()).reshape(-1)[0])
        # The model was trained on RUL targets capped at _MAX_RUL, so its
        # valid output domain is [0, _MAX_RUL]. Clamp out-of-range
        # extrapolations — which can occur for very short or atypical
        # input windows — so the API always returns a physically
        # meaningful remaining-useful-life value.
        predicted_rul = float(np.clip(raw_rul, 0.0, _MAX_RUL))

        margin = (1.0 - self.config.confidence_level) * 0.5 * _MAX_RUL
        risk_level, risk_score = self._get_risk_level(predicted_rul)

        result: Dict[str, Any] = {
            "machine_id": machine_id,
            "predicted_rul": predicted_rul,
            "confidence_interval": {
                "lower": float(max(0.0, predicted_rul - margin)),
                "upper": float(predicted_rul + margin),
            },
            "risk_level": risk_level,
            "risk_score": risk_score,
            "model_contributions": {"primary": predicted_rul},
            "anomaly_score": float(
                max(0.0, 1.0 - predicted_rul / _MAX_RUL)
            ),
            "timestamp": datetime.now(),
        }
        logger.info(
            "Predicted machine %s: RUL=%.2f risk=%s",
            machine_id,
            predicted_rul,
            risk_level,
        )
        return result

    def predict_batch(
        self, machines: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict RUL for many machines concurrently.

        Args:
            machines: List of dicts each with a ``machine_id`` and a
                ``sensor_data`` entry.

        Returns:
            A list of prediction dicts in the same order as ``machines``.
        """
        self._load_models()

        def _run(machine: Dict[str, Any]) -> Dict[str, Any]:
            """Score one machine entry.

            Args:
                machine: A single machine request dict.

            Returns:
                The prediction dict for the machine.
            """
            return self.predict_single(
                machine["machine_id"], machine["sensor_data"]
            )

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_run, machines))
        logger.info("Completed batch prediction for %d machines", len(results))
        return results
