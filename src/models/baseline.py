"""Tree-ensemble baseline RUL regressors with uncertainty estimation."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.utils.config import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_N_TRIALS = 50


class XGBoostRUL:
    """Gradient-boosted RUL regressor with Optuna-tuned hyperparameters.

    Point predictions come from a squared-error objective; predictive
    bounds come from a pair of quantile-regression models.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the regressor.

        Args:
            config: Shared model configuration.
        """
        self.config = config
        self.model: XGBRegressor = None
        self.lower_model: XGBRegressor = None
        self.upper_model: XGBRegressor = None
        self.best_params: Dict[str, Any] = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Tune hyperparameters and fit the point and quantile models.

        An Optuna study of 50 trials minimizes validation RMSE; the best
        parameters are then refit on the concatenated train and validation
        data, and matching 5%/95% quantile models are fitted for bounds.

        Args:
            X_train: Training feature matrix.
            y_train: Training RUL targets.
            X_val: Validation feature matrix.
            y_val: Validation RUL targets.
        """

        def objective(trial: optuna.Trial) -> float:
            """Return validation RMSE for one sampled parameter set.

            Args:
                trial: The Optuna trial supplying sampled parameters.

            Returns:
                The root mean squared error on the validation split.
            """
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
            }
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=_SEED,
                n_jobs=-1,
                **params,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return float(np.sqrt(mean_squared_error(y_val, preds)))

        sampler = optuna.samplers.TPESampler(seed=_SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=_N_TRIALS, show_progress_bar=False)

        self.best_params = study.best_params
        logger.info(
            "XGBoost tuning complete: best val RMSE=%.4f, params=%s",
            study.best_value,
            self.best_params,
        )

        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        self.model = XGBRegressor(
            objective="reg:squarederror",
            random_state=_SEED,
            n_jobs=-1,
            **self.best_params,
        )
        self.model.fit(X_full, y_full)

        self.lower_model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.05,
            random_state=_SEED,
            n_jobs=-1,
            **self.best_params,
        )
        self.lower_model.fit(X_full, y_full)

        self.upper_model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.95,
            random_state=_SEED,
            n_jobs=-1,
            **self.best_params,
        )
        self.upper_model.fit(X_full, y_full)
        logger.info("XGBoost point and quantile models fitted on full data")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict point RUL estimates.

        Args:
            X: Feature matrix.

        Returns:
            An array of predicted RUL values.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before predict().")
        return np.asarray(self.model.predict(X))

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict 5% and 95% quantile bounds for RUL.

        Args:
            X: Feature matrix.

        Returns:
            A tuple ``(lower, upper)`` of quantile prediction arrays.
        """
        if self.lower_model is None or self.upper_model is None:
            raise RuntimeError("Model must be trained before uncertainty.")
        lower = np.asarray(self.lower_model.predict(X))
        upper = np.asarray(self.upper_model.predict(X))
        return lower, upper

    def get_feature_importance(
        self, feature_names: List[str]
    ) -> pd.DataFrame:
        """Return feature importances sorted in descending order.

        Args:
            feature_names: Names aligned to the model's feature columns.

        Returns:
            A frame with ``feature`` and ``importance`` columns.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before importances.")
        importances = self.model.feature_importances_
        frame = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        return frame.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

    def save(self, path: Path) -> None:
        """Persist the fitted models and best parameters.

        Args:
            path: Destination file.
        """
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "lower_model": self.lower_model,
            "upper_model": self.upper_model,
            "best_params": self.best_params,
        }
        joblib.dump(state, path)
        logger.info("Saved XGBoostRUL to %s", path)

    def load(self, path: Path) -> None:
        """Restore fitted models and best parameters from disk.

        Args:
            path: Source file produced by :meth:`save`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"XGBoostRUL file not found: {path}")
        state = joblib.load(path)
        self.model = state["model"]
        self.lower_model = state["lower_model"]
        self.upper_model = state["upper_model"]
        self.best_params = state["best_params"]
        logger.info("Loaded XGBoostRUL from %s", path)


class RandomForestRUL:
    """Random-forest RUL regressor with tree-spread uncertainty."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the regressor.

        Args:
            config: Shared model configuration.
        """
        self.config = config
        self.model: RandomForestRegressor = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the random forest on training data.

        Args:
            X_train: Training feature matrix.
            y_train: Training RUL targets.
        """
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=_SEED,
            oob_score=True,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        logger.info(
            "RandomForestRUL fitted: oob_score=%.4f", self.model.oob_score_
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict point RUL estimates.

        Args:
            X: Feature matrix.

        Returns:
            An array of predicted RUL values.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before predict().")
        return np.asarray(self.model.predict(X))

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict RUL with per-sample tree-spread uncertainty.

        Uncertainty is the standard deviation of the individual tree
        predictions evaluated at each input row.

        Args:
            X: Feature matrix.

        Returns:
            A tuple ``(mean, std)`` of prediction arrays.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before uncertainty.")
        tree_preds = np.stack(
            [tree.predict(X) for tree in self.model.estimators_], axis=0
        )
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        return mean, std

    def save(self, path: Path) -> None:
        """Persist the fitted forest to disk.

        Args:
            path: Destination file.
        """
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved RandomForestRUL to %s", path)

    def load(self, path: Path) -> None:
        """Restore the fitted forest from disk.

        Args:
            path: Source file produced by :meth:`save`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"RandomForestRUL file not found: {path}")
        self.model = joblib.load(path)
        logger.info("Loaded RandomForestRUL from %s", path)
