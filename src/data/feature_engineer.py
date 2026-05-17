"""Time-series feature engineering for C-MAPSS sensor data."""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_ROLLING_WINDOWS: List[int] = [5, 10, 20, 30]
_EWMA_SPANS: List[int] = [10, 20]


class FeatureEngineer:
    """Derives rolling, derivative, smoothing and interaction features.

    All transformations respect engine boundaries by grouping on
    ``unit_id`` so that statistics never leak across engines.
    """

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        self.health_weights_: Dict[str, float] = {}

    @staticmethod
    def _sensor_columns(df: pd.DataFrame) -> List[str]:
        """Return the list of sensor feature columns in a frame.

        Args:
            df: Input frame.

        Returns:
            The list of ``sensor_*`` column names.
        """
        return [c for c in df.columns if c.startswith("sensor_")]

    def rolling_features(
        self, df: pd.DataFrame, windows: List[int] = _ROLLING_WINDOWS
    ) -> pd.DataFrame:
        """Compute per-engine rolling statistics for each sensor.

        For every sensor and window the mean, standard deviation, minimum,
        maximum and range (max minus min) are computed.

        Args:
            df: Frame with ``unit_id`` and sensor columns.
            windows: Rolling window lengths in cycles.

        Returns:
            A frame of rolling features aligned to ``df``'s index.
        """
        sensors = self._sensor_columns(df)
        grouped = df.groupby("unit_id")[sensors]
        features: Dict[str, pd.Series] = {}

        for window in windows:
            roll = grouped.rolling(window=window, min_periods=1)
            roll_mean = roll.mean().reset_index(level=0, drop=True)
            roll_std = roll.std().reset_index(level=0, drop=True).fillna(0.0)
            roll_min = roll.min().reset_index(level=0, drop=True)
            roll_max = roll.max().reset_index(level=0, drop=True)
            for sensor in sensors:
                features[f"{sensor}_roll{window}_mean"] = roll_mean[sensor]
                features[f"{sensor}_roll{window}_std"] = roll_std[sensor]
                features[f"{sensor}_roll{window}_min"] = roll_min[sensor]
                features[f"{sensor}_roll{window}_max"] = roll_max[sensor]
                features[f"{sensor}_roll{window}_range"] = (
                    roll_max[sensor] - roll_min[sensor]
                )

        return pd.DataFrame(features, index=df.index)

    def derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute first and second per-engine differences of each sensor.

        Args:
            df: Frame with ``unit_id`` and sensor columns.

        Returns:
            A frame of first- and second-derivative features.
        """
        sensors = self._sensor_columns(df)
        first = df.groupby("unit_id")[sensors].diff()
        second = first.groupby(df["unit_id"]).diff()

        first = first.fillna(0.0)
        second = second.fillna(0.0)
        first.columns = [f"{c}_d1" for c in sensors]
        second.columns = [f"{c}_d2" for c in sensors]
        return pd.concat([first, second], axis=1)

    def ewma(
        self, df: pd.DataFrame, spans: List[int] = _EWMA_SPANS
    ) -> pd.DataFrame:
        """Compute per-engine exponentially weighted moving averages.

        Args:
            df: Frame with ``unit_id`` and sensor columns.
            spans: EWMA spans in cycles.

        Returns:
            A frame of EWMA features aligned to ``df``'s index.
        """
        sensors = self._sensor_columns(df)
        features: Dict[str, pd.Series] = {}

        for span in spans:
            ewm = (
                df.groupby("unit_id")[sensors]
                .apply(lambda g: g.ewm(span=span, adjust=False).mean())
                .reset_index(level=0, drop=True)
            )
            for sensor in sensors:
                features[f"{sensor}_ewma{span}"] = ewm[sensor]

        return pd.DataFrame(features, index=df.index)

    def interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute physically meaningful sensor-ratio interaction features.

        Args:
            df: Frame containing the required sensor columns.

        Returns:
            A frame with the ratio interaction features.
        """
        eps = 1e-8
        features = pd.DataFrame(index=df.index)
        features["t30_t50_ratio"] = df["sensor_7"] / (df["sensor_12"] + eps)
        features["p30_p2_ratio"] = df["sensor_14"] / (df["sensor_2"] + eps)
        return features

    def health_index(self, df: pd.DataFrame) -> pd.Series:
        """Compute a correlation-weighted health index from sensors.

        Each sensor's Pearson correlation with RUL determines its weight;
        weights are the normalized absolute correlations. Fitted weights
        are cached in :attr:`health_weights_`.

        Args:
            df: Frame with sensor columns and a ``rul`` column.

        Returns:
            A Series with the per-row health index.
        """
        sensors = self._sensor_columns(df)
        correlations = {
            sensor: df[sensor].corr(df["rul"]) for sensor in sensors
        }
        abs_corr = {s: abs(c) for s, c in correlations.items() if not np.isnan(c)}
        total = sum(abs_corr.values())

        if total == 0.0:
            self.health_weights_ = {s: 0.0 for s in sensors}
            return pd.Series(0.0, index=df.index, name="health_index")

        self.health_weights_ = {s: w / total for s, w in abs_corr.items()}
        index = pd.Series(0.0, index=df.index, name="health_index")
        for sensor, weight in self.health_weights_.items():
            index = index + df[sensor] * weight
        return index

    @staticmethod
    def cycle_norm(df: pd.DataFrame) -> pd.Series:
        """Compute the per-engine normalized life fraction.

        Args:
            df: Frame with ``unit_id`` and ``cycle`` columns.

        Returns:
            A Series of cycle divided by the engine's maximum cycle.
        """
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        norm = df["cycle"] / max_cycle
        norm.name = "cycle_norm"
        return norm

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply every feature transformation and drop NaN rolling rows.

        Args:
            df: Frame with ``unit_id``, ``cycle``, sensor and ``rul`` columns.

        Returns:
            The augmented frame with NaN-bearing rows removed.
        """
        rolling = self.rolling_features(df)
        derivative = self.derivatives(df)
        ewma = self.ewma(df)
        interaction = self.interactions(df)

        out = pd.concat([df, rolling, derivative, ewma, interaction], axis=1)
        out["health_index"] = self.health_index(df)
        out["cycle_norm"] = self.cycle_norm(df)

        before = len(out)
        out = out.dropna().reset_index(drop=True)
        logger.info(
            "Feature engineering produced %d columns; dropped %d NaN rows",
            out.shape[1],
            before - len(out),
        )
        return out
