"""Loading and engine-wise splitting of the C-MAPSS turbofan dataset."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_RAW_DATA_DIR = Path("data/raw")


class CMAPSSLoader:
    """Loader for the NASA C-MAPSS turbofan degradation dataset.

    Reads the raw whitespace-delimited ``train``/``test``/``RUL`` text files,
    assigns canonical column names, drops uninformative sensor channels and
    derives Remaining Useful Life (RUL) targets.
    """

    COLUMNS: List[str] = (
        ["unit_id", "cycle", "op_1", "op_2", "op_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    SENSORS_DROP: List[int] = [1, 5, 6, 10, 16, 18, 19]
    RUL_CLIP: int = 125

    def __init__(self, raw_data_dir: Path = _RAW_DATA_DIR) -> None:
        """Initialize the loader.

        Args:
            raw_data_dir: Directory containing the raw C-MAPSS text files.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self._drop_cols: List[str] = [f"sensor_{i}" for i in self.SENSORS_DROP]

    def _read_table(self, path: Path) -> pd.DataFrame:
        """Read a single whitespace-delimited C-MAPSS file.

        Args:
            path: Path to the ``train``/``test`` text file.

        Returns:
            A DataFrame with canonical column names and dropped sensors.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Expected C-MAPSS file not found: {path}")
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        df = df.iloc[:, : len(self.COLUMNS)]
        df.columns = self.COLUMNS
        df = df.drop(columns=self._drop_cols)
        return df

    def load_subset(
        self, subset: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load a single C-MAPSS subset (e.g. ``FD001``).

        The train frame receives a clipped RUL target computed as the
        per-engine maximum cycle minus the current cycle. The test frame
        receives RUL values derived from the accompanying ``RUL`` file:
        the published value is the RUL at the last observed cycle, so RUL
        at each earlier cycle is offset accordingly.

        Args:
            subset: Subset identifier, one of ``FD001``..``FD004``.

        Returns:
            A tuple ``(train_df, test_df, rul_series)`` where ``rul_series``
            is the raw per-engine RUL from ``RUL_<subset>.txt``.
        """
        subset = subset.upper()
        train_path = self.raw_data_dir / f"train_{subset}.txt"
        test_path = self.raw_data_dir / f"test_{subset}.txt"
        rul_path = self.raw_data_dir / f"RUL_{subset}.txt"

        train_df = self._read_table(train_path)
        test_df = self._read_table(test_path)

        if not rul_path.is_file():
            raise FileNotFoundError(f"Expected RUL file not found: {rul_path}")
        rul_series = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0]
        rul_series.name = "rul"

        train_df = self._add_train_rul(train_df)
        test_df = self._add_test_rul(test_df, rul_series)

        logger.info(
            "Loaded subset %s: train=%d rows, test=%d rows, %d engines (test)",
            subset,
            len(train_df),
            len(test_df),
            test_df["unit_id"].nunique(),
        )
        return train_df, test_df, rul_series

    def _add_train_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach a clipped RUL target to a training frame.

        Args:
            df: Training frame with ``unit_id`` and ``cycle`` columns.

        Returns:
            The frame with an added ``rul`` column.
        """
        df = df.copy()
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        df["rul"] = (max_cycle - df["cycle"]).clip(upper=self.RUL_CLIP)
        return df

    def _add_test_rul(
        self, df: pd.DataFrame, rul_series: pd.Series
    ) -> pd.DataFrame:
        """Attach a clipped RUL target to a test frame.

        Args:
            df: Test frame with ``unit_id`` and ``cycle`` columns.
            rul_series: Per-engine RUL at the last observed cycle.

        Returns:
            The frame with an added ``rul`` column.
        """
        df = df.copy()
        rul_map = {
            unit_id: int(rul_series.iloc[i])
            for i, unit_id in enumerate(sorted(df["unit_id"].unique()))
        }
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        final_rul = df["unit_id"].map(rul_map)
        df["rul"] = (final_rul + (max_cycle - df["cycle"])).clip(
            upper=self.RUL_CLIP
        )
        return df

    def load_all(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
        """Load every C-MAPSS subset.

        Returns:
            A dict keyed by ``FD001``..``FD004`` mapping to the tuple
            returned by :meth:`load_subset`.
        """
        subsets = ["FD001", "FD002", "FD003", "FD004"]
        return {subset: self.load_subset(subset) for subset in subsets}

    @staticmethod
    def split_by_engine(
        df: pd.DataFrame, train_r: float = 0.8, val_r: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a frame into train/val/test with no engine-level leakage.

        Engine identifiers are shuffled with a fixed seed and partitioned;
        every row of a given engine lands in exactly one split.

        Args:
            df: Frame containing a ``unit_id`` column.
            train_r: Fraction of engines assigned to the train split.
            val_r: Fraction of engines assigned to the validation split.

        Returns:
            A tuple ``(train_df, val_df, test_df)``.
        """
        unit_ids = np.array(sorted(df["unit_id"].unique()))
        rng = np.random.default_rng(_SEED)
        rng.shuffle(unit_ids)

        n_units = len(unit_ids)
        n_train = int(round(n_units * train_r))
        n_val = int(round(n_units * val_r))

        train_ids = set(unit_ids[:n_train])
        val_ids = set(unit_ids[n_train : n_train + n_val])
        test_ids = set(unit_ids[n_train + n_val :])

        train_df = df[df["unit_id"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["unit_id"].isin(val_ids)].reset_index(drop=True)
        test_df = df[df["unit_id"].isin(test_ids)].reset_index(drop=True)

        logger.info(
            "Engine split: train=%d, val=%d, test=%d engines",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )
        return train_df, val_df, test_df
