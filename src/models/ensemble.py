"""Stacking ensemble combining deep and tree-based RUL regressors."""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_BASE_ORDER: List[str] = ["lstm", "tcn", "transformer", "xgboost", "rf"]


class StackingEnsemble:
    """Stacking ensemble of base RUL models with a meta-learner.

    Five heterogeneous base models (LSTM, TCN, transformer, XGBoost and
    random forest) produce predictions that are blended in three ways: a
    Ridge meta-learner trained on out-of-fold predictions, an optimized
    convex weight vector, and conformal calibration for prediction
    intervals.
    """

    def __init__(self, models: Dict[str, Any], config: ModelConfig) -> None:
        """Initialize the ensemble.

        Args:
            models: Mapping of base-model name to model instance. Expected
                keys are ``lstm``, ``tcn``, ``transformer``, ``xgboost``
                and ``rf``.
            config: Shared model configuration.
        """
        self.models = models
        self.config = config
        self.meta_learner: Ridge = Ridge(alpha=1.0, random_state=_SEED)
        self.weights: np.ndarray = np.full(
            len(_BASE_ORDER), 1.0 / len(_BASE_ORDER)
        )
        self.calibration_residuals: np.ndarray = np.array([])
        self._meta_fitted: bool = False

    def _is_torch_model(self, name: str) -> bool:
        """Return whether a named base model is a PyTorch module.

        Args:
            name: Base-model key.

        Returns:
            ``True`` for the neural base models, ``False`` otherwise.
        """
        return name in {"lstm", "tcn", "transformer"}

    def _predict_base(self, name: str, X: np.ndarray) -> np.ndarray:
        """Run a single base model and return point predictions.

        Args:
            name: Base-model key.
            X: Feature array. For torch models this is windowed data of
                shape ``(n, window, features)``; for tree models a 2-D
                feature matrix.

        Returns:
            A 1-D array of predicted RUL values.
        """
        model = self.models[name]
        if self._is_torch_model(name):
            model.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(X, dtype=torch.float32)
                output = model(tensor)
                if isinstance(output, tuple):
                    output = output[0]
            return np.asarray(output.cpu().numpy()).reshape(-1)
        return np.asarray(model.predict(X)).reshape(-1)

    def get_oof_predictions(
        self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5
    ) -> np.ndarray:
        """Generate out-of-fold predictions from every base model.

        Each base model is refit on every training fold and used to
        predict the held-out fold, yielding leakage-free meta features.

        Args:
            X_train: Training features indexable along the first axis.
            y_train: Training RUL targets.
            cv: Number of cross-validation folds.

        Returns:
            An array of shape ``(n_samples, n_base_models)`` of OOF
            predictions ordered by :data:`_BASE_ORDER`.
        """
        n_samples = len(y_train)
        oof = np.zeros((n_samples, len(_BASE_ORDER)))
        splitter = KFold(n_splits=cv, shuffle=True, random_state=_SEED)

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X_train)):
            for col, name in enumerate(_BASE_ORDER):
                model = self.models[name]
                if self._is_torch_model(name):
                    oof[val_idx, col] = self._predict_base(
                        name, X_train[val_idx]
                    )
                    continue
                model.fit(X_train[train_idx], y_train[train_idx])
                oof[val_idx, col] = np.asarray(
                    model.predict(X_train[val_idx])
                ).reshape(-1)
            logger.info("Computed OOF predictions for fold %d", fold + 1)

        logger.info("Out-of-fold prediction matrix shape: %s", oof.shape)
        return oof

    def fit_meta_learner(
        self, oof_predictions: np.ndarray, y_train: np.ndarray
    ) -> None:
        """Fit the Ridge meta-learner on out-of-fold predictions.

        Args:
            oof_predictions: OOF prediction matrix from
                :meth:`get_oof_predictions`.
            y_train: Training RUL targets aligned to the OOF rows.
        """
        self.meta_learner.fit(oof_predictions, y_train)
        self._meta_fitted = True
        logger.info(
            "Fitted Ridge meta-learner with coefficients %s",
            np.round(self.meta_learner.coef_, 4),
        )

    def optimize_weights(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Optimize convex blend weights minimizing validation RMSE.

        The weights are constrained to be non-negative and sum to one;
        :func:`scipy.optimize.minimize` searches the simplex with SLSQP.

        Args:
            X_val: Validation features indexable along the first axis.
            y_val: Validation RUL targets.

        Returns:
            The optimized weight vector ordered by :data:`_BASE_ORDER`.
        """
        base_preds = np.column_stack(
            [self._predict_base(name, X_val) for name in _BASE_ORDER]
        )

        def objective(weights: np.ndarray) -> float:
            """Return the validation RMSE for a weight vector.

            Args:
                weights: Candidate blend weights.

            Returns:
                The root mean squared error of the weighted blend.
            """
            blended = base_preds @ weights
            return float(np.sqrt(np.mean((blended - y_val) ** 2)))

        n_models = len(_BASE_ORDER)
        constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
        bounds = [(0.0, 1.0)] * n_models
        initial = np.full(n_models, 1.0 / n_models)

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        self.weights = np.asarray(result.x)
        logger.info(
            "Optimized ensemble weights %s (val RMSE=%.4f)",
            np.round(self.weights, 4),
            result.fun,
        )
        return self.weights

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Store absolute residuals for conformal prediction intervals.

        Args:
            X_cal: Calibration features indexable along the first axis.
            y_cal: Calibration RUL targets.
        """
        base_preds = np.column_stack(
            [self._predict_base(name, X_cal) for name in _BASE_ORDER]
        )
        blended = base_preds @ self.weights
        self.calibration_residuals = np.abs(y_cal - blended)
        logger.info(
            "Stored %d calibration residuals",
            self.calibration_residuals.size,
        )

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict RUL for a single input with blended uncertainty.

        Args:
            X: Features for one sample, indexable along the first axis.

        Returns:
            A dict with the meta-learner ``rul_prediction``, conformal
            ``lower_bound`` / ``upper_bound``, per-model
            ``model_contributions``, an ``anomaly_score`` and the convex
            ``weighted_prediction``.
        """
        base_preds = np.column_stack(
            [self._predict_base(name, X) for name in _BASE_ORDER]
        )

        weighted = float((base_preds @ self.weights).mean())
        if self._meta_fitted:
            meta_pred = float(self.meta_learner.predict(base_preds).mean())
        else:
            meta_pred = weighted

        if self.calibration_residuals.size > 0:
            quantile = float(
                np.quantile(self.calibration_residuals, 0.90)
            )
        else:
            quantile = 0.0

        contributions = {
            name: float(base_preds[:, col].mean() * self.weights[col])
            for col, name in enumerate(_BASE_ORDER)
        }

        anomaly_score = 0.0
        if "rf" in self.models:
            spread = base_preds.std(axis=1).mean()
            anomaly_score = float(spread)

        logger.info(
            "Ensemble prediction: meta=%.4f weighted=%.4f", meta_pred, weighted
        )
        return {
            "rul_prediction": meta_pred,
            "lower_bound": meta_pred - quantile,
            "upper_bound": meta_pred + quantile,
            "model_contributions": contributions,
            "anomaly_score": anomaly_score,
            "weighted_prediction": weighted,
        }

    def save(self, path: Path) -> None:
        """Persist the meta-learner, weights and calibration residuals.

        Args:
            path: Destination file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "meta_learner": self.meta_learner,
            "weights": self.weights,
            "calibration_residuals": self.calibration_residuals,
            "meta_fitted": self._meta_fitted,
        }
        joblib.dump(state, path)
        logger.info("Saved StackingEnsemble state to %s", path)

    def load(self, path: Path) -> None:
        """Restore the meta-learner, weights and calibration residuals.

        Args:
            path: Source file produced by :meth:`save`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Ensemble state file not found: {path}")
        state = joblib.load(path)
        self.meta_learner = state["meta_learner"]
        self.weights = state["weights"]
        self.calibration_residuals = state["calibration_residuals"]
        self._meta_fitted = state["meta_fitted"]
        logger.info("Loaded StackingEnsemble state from %s", path)
