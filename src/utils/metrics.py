"""Regression and prognostics evaluation metrics for RUL prediction."""

from typing import Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, list, pd.Series]


def _as_array(values: ArrayLike) -> np.ndarray:
    """Coerce an array-like input into a 1-D float ``numpy`` array.

    Args:
        values: Sequence of numeric values.

    Returns:
        A flattened float64 ``numpy`` array.
    """
    return np.asarray(values, dtype=np.float64).ravel()


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the root mean squared error.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values.

    Returns:
        The root mean squared error.
    """
    yt, yp = _as_array(y_true), _as_array(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean absolute error.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values.

    Returns:
        The mean absolute error.
    """
    yt, yp = _as_array(y_true), _as_array(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the coefficient of determination (R^2).

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted values.

    Returns:
        The R^2 score; returns 0.0 when the total sum of squares is zero.
    """
    yt, yp = _as_array(y_true), _as_array(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def score_s(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the PHM Challenge asymmetric scoring function.

    For each sample with error ``d = y_pred - y_true`` the score is
    ``exp(-d / 13) - 1`` when ``d < 0`` (early prediction) and
    ``exp(d / 10) - 1`` otherwise (late prediction). The total is the
    sum over all samples, penalizing late predictions more heavily.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        The summed asymmetric score.
    """
    yt, yp = _as_array(y_true), _as_array(y_pred)
    d = yp - yt
    penalties = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalties))


def accuracy_at_k(y_true: ArrayLike, y_pred: ArrayLike, k: int) -> float:
    """Compute the fraction of predictions within +/- ``k`` cycles.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.
        k: Tolerance in cycles.

    Returns:
        The percentage (0-100) of predictions within the tolerance band.
    """
    yt, yp = _as_array(y_true), _as_array(y_pred)
    if yt.size == 0:
        return 0.0
    within = np.abs(yp - yt) <= k
    return float(np.mean(within) * 100.0)


def calibration_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> float:
    """Compute the absolute calibration error of prediction intervals.

    The empirical coverage is the fraction of ground-truth values that
    fall within the ``[lower, upper]`` interval. This is compared with
    the implied nominal coverage; the returned value is the absolute
    difference between empirical coverage and that nominal target.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted point estimates (used for nominal inference).
        lower: Lower bounds of the prediction intervals.
        upper: Upper bounds of the prediction intervals.

    Returns:
        The absolute calibration error in the range [0, 1].
    """
    yt = _as_array(y_true)
    lo, up = _as_array(lower), _as_array(upper)
    if yt.size == 0:
        return 0.0
    covered = (yt >= lo) & (yt <= up)
    empirical_coverage = float(np.mean(covered))
    nominal_coverage = float(np.mean((_as_array(y_pred) >= lo) & (_as_array(y_pred) <= up)))
    return float(abs(empirical_coverage - nominal_coverage))


def per_engine_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped RUL error statistics per engine unit.

    The input frame must contain the columns ``engine_id``, ``y_true``
    and ``y_pred``. For each engine the RMSE, MAE, mean signed error,
    PHM score and sample count are computed.

    Args:
        df: DataFrame with ``engine_id``, ``y_true`` and ``y_pred`` columns.

    Returns:
        A DataFrame indexed by ``engine_id`` with one row of error
        statistics per engine.
    """
    required = {"engine_id", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    records = []
    for engine_id, group in df.groupby("engine_id"):
        yt = group["y_true"].to_numpy(dtype=np.float64)
        yp = group["y_pred"].to_numpy(dtype=np.float64)
        records.append(
            {
                "engine_id": engine_id,
                "n_samples": int(yt.size),
                "rmse": rmse(yt, yp),
                "mae": mae(yt, yp),
                "mean_error": float(np.mean(yp - yt)),
                "score_s": score_s(yt, yp),
            }
        )

    return pd.DataFrame.from_records(records).set_index("engine_id")
