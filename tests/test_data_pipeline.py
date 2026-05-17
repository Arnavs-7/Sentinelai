"""Tests for the C-MAPSS data loading, feature and windowing pipeline."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import FeatureEngineer
from src.data.loader import CMAPSSLoader
from src.data.windowing import SlidingWindowProcessor

_FD001_TRAIN = Path("data/raw/train_FD001.txt")
_SENSOR_IDS: List[int] = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14]


def _synthetic_frame(n_engines: int = 3, n_cycles: int = 50) -> pd.DataFrame:
    """Build a synthetic multi-engine sensor frame.

    Args:
        n_engines: Number of distinct engines.
        n_cycles: Number of cycles per engine.

    Returns:
        A frame with ``unit_id``, ``cycle``, sensor and ``rul`` columns.
    """
    rng = np.random.default_rng(42)
    rows = []
    for unit_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            record = {"unit_id": unit_id, "cycle": cycle}
            for sensor in _SENSOR_IDS:
                record[f"sensor_{sensor}"] = float(
                    rng.normal(100.0, 5.0) + cycle * 0.1
                )
            record["rul"] = float(n_cycles - cycle)
            rows.append(record)
    return pd.DataFrame(rows)


@pytest.mark.skipif(
    not _FD001_TRAIN.is_file(), reason="raw FD001 data not available"
)
def test_loader_columns() -> None:
    """The loaded FD001 subset has canonical columns and dropped sensors."""
    train_df, test_df, rul_series = CMAPSSLoader().load_subset("FD001")
    assert "unit_id" in train_df.columns
    assert "cycle" in train_df.columns
    assert "rul" in train_df.columns
    for dropped in CMAPSSLoader.SENSORS_DROP:
        assert f"sensor_{dropped}" not in train_df.columns
    assert not test_df.empty
    assert not rul_series.empty


def test_rul_calculation() -> None:
    """Train RUL equals max cycle minus cycle, clipped at 125."""
    frame = pd.DataFrame(
        {
            "unit_id": [1] * 200 + [2] * 80,
            "cycle": list(range(1, 201)) + list(range(1, 81)),
        }
    )
    result = CMAPSSLoader()._add_train_rul(frame)
    assert result["rul"].max() <= CMAPSSLoader.RUL_CLIP
    engine_one = result[result["unit_id"] == 1]
    assert engine_one[engine_one["cycle"] == 200]["rul"].iloc[0] == 0
    assert engine_one[engine_one["cycle"] == 100]["rul"].iloc[0] == 100
    assert engine_one[engine_one["cycle"] == 1]["rul"].iloc[0] == 125
    engine_two = result[result["unit_id"] == 2]
    assert engine_two[engine_two["cycle"] == 80]["rul"].iloc[0] == 0
    assert engine_two[engine_two["cycle"] == 1]["rul"].iloc[0] == 79


def test_no_data_leakage() -> None:
    """Train, val and test splits never share an engine identifier."""
    frame = _synthetic_frame(n_engines=20, n_cycles=30)
    train_df, val_df, test_df = CMAPSSLoader.split_by_engine(frame)

    train_ids = set(train_df["unit_id"].unique())
    val_ids = set(val_df["unit_id"].unique())
    test_ids = set(test_df["unit_id"].unique())

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(frame["unit_id"].unique())


def test_feature_engineering_shape() -> None:
    """Feature engineering produces strictly more columns than the input."""
    frame = _synthetic_frame()
    engineered = FeatureEngineer().fit_transform(frame)
    assert engineered.shape[1] > frame.shape[1]
    assert not engineered.empty


def test_windowing_shape() -> None:
    """Windowing yields tensors of shape (n, 30, n_features) and (n,)."""
    frame = _synthetic_frame(n_engines=2, n_cycles=50)
    feature_cols = [f"sensor_{s}" for s in _SENSOR_IDS]
    X, y = SlidingWindowProcessor().create_windows(
        frame, feature_cols=feature_cols, target_col="rul", window_size=30
    )
    expected_windows = 2 * (50 - 30 + 1)
    assert tuple(X.shape) == (expected_windows, 30, len(feature_cols))
    assert tuple(y.shape) == (expected_windows,)


def test_windowing_target() -> None:
    """Each window's target is the RUL at its final cycle."""
    rul_values = list(range(100, 140))
    frame = pd.DataFrame(
        {
            "unit_id": [1] * 40,
            "cycle": list(range(1, 41)),
            "sensor_2": np.arange(40, dtype=float),
            "rul": rul_values,
        }
    )
    X, y = SlidingWindowProcessor().create_windows(
        frame, feature_cols=["sensor_2"], target_col="rul", window_size=30
    )
    assert pytest.approx(float(y[0])) == rul_values[29]
    assert pytest.approx(float(y[-1])) == rul_values[39]
