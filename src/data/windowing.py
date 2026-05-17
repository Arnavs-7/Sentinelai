"""Sliding-window tensor construction for sequence models."""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SlidingWindowProcessor:
    """Builds fixed-length sliding windows from per-engine time series.

    Each window is a contiguous slice of cycles from a single engine; the
    regression target is the RUL recorded at the window's final step.
    """

    def create_windows(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "rul",
        window_size: int = 30,
        stride: int = 1,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Slide a fixed-length window over each engine's sequence.

        Engines shorter than ``window_size`` contribute no windows.

        Args:
            df: Frame with ``unit_id``, feature and target columns, ordered
                by cycle within each engine.
            feature_cols: Columns to stack into the feature dimension.
            target_col: Column holding the regression target.
            window_size: Number of cycles per window.
            stride: Step between consecutive window start positions.

        Returns:
            A tuple ``(X, y)`` of float tensors with shapes
            ``(n_windows, window_size, n_features)`` and ``(n_windows,)``.
        """
        x_chunks: List[np.ndarray] = []
        y_values: List[float] = []

        for unit_id, group in df.groupby("unit_id"):
            group = group.sort_values("cycle")
            features = group[feature_cols].to_numpy(dtype=np.float32)
            targets = group[target_col].to_numpy(dtype=np.float32)
            n_rows = len(group)

            if n_rows < window_size:
                logger.debug(
                    "Engine %s has %d rows < window_size %d; skipped",
                    unit_id,
                    n_rows,
                    window_size,
                )
                continue

            for start in range(0, n_rows - window_size + 1, stride):
                end = start + window_size
                x_chunks.append(features[start:end])
                y_values.append(float(targets[end - 1]))

        if not x_chunks:
            n_features = len(feature_cols)
            logger.warning("No windows produced; returning empty tensors")
            return (
                torch.empty((0, window_size, n_features), dtype=torch.float32),
                torch.empty((0,), dtype=torch.float32),
            )

        x_array = np.stack(x_chunks).astype(np.float32)
        y_array = np.asarray(y_values, dtype=np.float32)

        x_tensor = torch.from_numpy(x_array).float()
        y_tensor = torch.from_numpy(y_array).float()

        logger.info(
            "Created %d windows of shape (%d, %d)",
            x_tensor.shape[0],
            window_size,
            len(feature_cols),
        )
        return x_tensor, y_tensor
