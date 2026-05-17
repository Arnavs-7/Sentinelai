"""Fitting and applying scaling and operating-condition clustering."""

from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_OP_COLUMNS: List[str] = ["op_1", "op_2", "op_3"]
_NON_FEATURE_COLUMNS: List[str] = ["unit_id", "cycle", "rul", "op_cluster"]


class CMAPSSPreprocessor:
    """Scales sensor channels and clusters operating conditions.

    A :class:`~sklearn.preprocessing.MinMaxScaler` is fitted per sensor
    column on the training data only, and a
    :class:`~sklearn.cluster.KMeans` model assigns each row to one of six
    operating-condition regimes.
    """

    N_CLUSTERS: int = 6

    def __init__(self) -> None:
        """Initialize an unfitted preprocessor."""
        self.scaler: Optional[MinMaxScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.sensor_columns: List[str] = []
        self._fitted: bool = False

    @staticmethod
    def _sensor_columns(df: pd.DataFrame) -> List[str]:
        """Return the list of sensor feature columns in a frame.

        Args:
            df: Input frame.

        Returns:
            The sorted list of ``sensor_*`` column names.
        """
        return [c for c in df.columns if c.startswith("sensor_")]

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the scaler and clustering model on training data.

        Args:
            train_df: Training frame with sensor and operating columns.
        """
        self.sensor_columns = self._sensor_columns(train_df)

        self.scaler = MinMaxScaler()
        self.scaler.fit(train_df[self.sensor_columns])

        self.kmeans = KMeans(n_clusters=self.N_CLUSTERS, random_state=_SEED, n_init=10)
        self.kmeans.fit(train_df[_OP_COLUMNS])

        self._fitted = True
        logger.info(
            "Fitted preprocessor on %d sensors, %d operating-condition clusters",
            len(self.sensor_columns),
            self.N_CLUSTERS,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler and cluster assignment to a frame.

        Args:
            df: Frame with the same sensor/operating columns seen at fit time.

        Returns:
            A new frame with scaled sensors and an added ``op_cluster`` column.
        """
        if not self._fitted or self.scaler is None or self.kmeans is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        out = df.copy()
        out[self.sensor_columns] = self.scaler.transform(out[self.sensor_columns])
        # KMeans was fitted on float64 data, so its Cython prediction
        # routine rejects a float32 input (as produced by the inference
        # path). Cast the operating columns to the fitted centers' dtype
        # so clustering works regardless of the caller's dtype.
        op_values = out[_OP_COLUMNS].astype(
            self.kmeans.cluster_centers_.dtype
        )
        out["op_cluster"] = self.kmeans.predict(op_values).astype(int)
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on the training frame and return its transformed version.

        Args:
            train_df: Training frame.

        Returns:
            The transformed training frame.
        """
        self.fit(train_df)
        return self.transform(train_df)

    def save(self, path: Path) -> None:
        """Persist the fitted preprocessor state to disk.

        Args:
            path: Destination file path.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "scaler": self.scaler,
            "kmeans": self.kmeans,
            "sensor_columns": self.sensor_columns,
        }
        joblib.dump(state, path)
        logger.info("Saved preprocessor to %s", path)

    def load(self, path: Path) -> None:
        """Restore preprocessor state from disk.

        Args:
            path: Source file path produced by :meth:`save`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        state = joblib.load(path)
        self.scaler = state["scaler"]
        self.kmeans = state["kmeans"]
        self.sensor_columns = state["sensor_columns"]
        self._fitted = True
        logger.info("Loaded preprocessor from %s", path)
