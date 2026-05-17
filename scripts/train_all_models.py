"""Sequential end-to-end training of every SentinelAI RUL model.

The script loads the C-MAPSS data (falling back to generated demo data),
runs the full preprocessing and feature-engineering pipeline, trains the
deep sequence models with the :class:`Trainer` and MLflow tracking,
trains the tree-based baselines, prints comparison tables and saves a
production model bundle.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import CMAPSSDataset, get_dataloader
from src.data.feature_engineer import FeatureEngineer
from src.data.loader import CMAPSSLoader
from src.data.preprocessor import CMAPSSPreprocessor
from src.data.windowing import SlidingWindowProcessor
from src.models.baseline import RandomForestRUL, XGBoostRUL
from src.models.lstm_model import get_lstm_model
from src.models.model_utils import save_model, set_seed
from src.models.tcn_model import get_tcn_model
from src.models.transformer_model import get_transformer_model
from src.training.losses import RMSELoss
from src.training.trainer import Trainer
from src.utils.config import DataConfig, ModelConfig, TrainingConfig
from src.utils.logger import get_logger
from src.utils.metrics import accuracy_at_k, mae, r2, rmse, score_s

logger = get_logger(__name__)

_SEED = 42
_SUBSET = "FD001"
_WINDOW_SIZE = 30
_CHECKPOINT_DIR = Path("models_saved")
_DEMO_SENSORS = Path("data/processed/demo_sensors.csv")
_PRODUCTION_PATH = _CHECKPOINT_DIR / "production_model.pkl"
_PREPROCESSOR_PATH = _CHECKPOINT_DIR / "preprocessor.joblib"
_BEST_MODEL_PATH = _CHECKPOINT_DIR / "best_model.pt"
_NON_FEATURE = {"unit_id", "cycle", "rul", "machine_id"}


def _load_raw_frame() -> pd.DataFrame:
    """Load a single training frame from raw C-MAPSS data or demo data.

    Returns:
        A frame with ``unit_id``, ``cycle``, operating, sensor and ``rul``
        columns suitable for the preprocessing pipeline.
    """
    try:
        loader = CMAPSSLoader()
        train_df, _, _ = loader.load_subset(_SUBSET)
        logger.info("Loaded raw C-MAPSS subset %s (%d rows)", _SUBSET, len(train_df))
        return train_df
    except FileNotFoundError:
        logger.warning("Raw C-MAPSS data not found; falling back to demo data")

    if not _DEMO_SENSORS.is_file():
        raise FileNotFoundError(
            "Neither raw C-MAPSS data nor demo data is available. "
            "Run scripts/generate_demo_data.py first."
        )

    frame = pd.read_csv(_DEMO_SENSORS)
    frame = frame.rename(columns={"machine_id": "unit_id"})
    rng = np.random.default_rng(_SEED)
    for column in ("op_1", "op_2", "op_3"):
        frame[column] = rng.normal(0.0, 1.0, size=len(frame))
    logger.info("Loaded demo training frame (%d rows)", len(frame))
    return frame


def _build_datasets() -> Tuple[
    Dict[str, "torch.utils.data.DataLoader"],
    Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    List[str],
]:
    """Run the full pipeline and build windowed datasets and loaders.

    Returns:
        A tuple ``(loaders, tensors, feature_cols)`` where ``loaders`` maps
        split names to dataloaders, ``tensors`` maps split names to
        ``(X, y)`` tensor tuples and ``feature_cols`` lists the feature
        column names.
    """
    frame = _load_raw_frame()
    train_df, val_df, test_df = CMAPSSLoader.split_by_engine(frame)

    preprocessor = CMAPSSPreprocessor()
    train_scaled = preprocessor.fit_transform(train_df)
    val_scaled = preprocessor.transform(val_df)
    test_scaled = preprocessor.transform(test_df)
    preprocessor.save(_PREPROCESSOR_PATH)

    engineer = FeatureEngineer()
    train_feat = engineer.fit_transform(train_scaled)
    val_feat = engineer.fit_transform(val_scaled)
    test_feat = engineer.fit_transform(test_scaled)

    feature_cols = [c for c in train_feat.columns if c not in _NON_FEATURE]
    windower = SlidingWindowProcessor()

    loaders: Dict[str, "torch.utils.data.DataLoader"] = {}
    tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, feat, shuffle in (
        ("train", train_feat, True),
        ("val", val_feat, False),
        ("test", test_feat, False),
    ):
        X, y = windower.create_windows(
            feat,
            feature_cols=feature_cols,
            target_col="rul",
            window_size=_WINDOW_SIZE,
            stride=1,
        )
        tensors[name] = (X, y)
        dataset = CMAPSSDataset(X, y, mode=name)
        loaders[name] = get_dataloader(
            dataset, batch_size=TrainingConfig().batch_size, shuffle=shuffle
        )
        logger.info("Built %s dataset: %s windows", name, tuple(X.shape))

    return loaders, tensors, feature_cols


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the standard RUL metric suite for a set of predictions.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.

    Returns:
        A dict of metric name to value.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "score_s": score_s(y_true, y_pred),
        "acc_10": accuracy_at_k(y_true, y_pred, 10),
        "acc_15": accuracy_at_k(y_true, y_pred, 15),
    }


def _torch_predict(
    model: torch.nn.Module, X: torch.Tensor
) -> np.ndarray:
    """Run a torch model over a feature tensor and return predictions.

    Args:
        model: The trained model.
        X: Windowed feature tensor.

    Returns:
        A 1-D array of predicted RUL values.
    """
    model.eval()
    with torch.no_grad():
        output = model(X)
        if isinstance(output, tuple):
            output = output[0]
    return np.asarray(output.cpu().numpy()).reshape(-1)


def _train_torch_models(
    loaders: Dict[str, "torch.utils.data.DataLoader"],
    tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    input_size: int,
    epochs: int,
) -> Tuple[Dict[str, torch.nn.Module], List[Dict[str, object]]]:
    """Train the LSTM, TCN and transformer models with the Trainer.

    Args:
        loaders: Mapping of split name to dataloader.
        tensors: Mapping of split name to ``(X, y)`` tensors.
        input_size: Number of input features per time step.
        epochs: Maximum number of training epochs per model.

    Returns:
        A tuple ``(models, rows)`` of trained models and progress rows.
    """
    model_config = ModelConfig()
    train_config = TrainingConfig()
    train_config.max_epochs = epochs
    builders = {
        "lstm": get_lstm_model,
        "tcn": get_tcn_model,
        "transformer": get_transformer_model,
    }

    models: Dict[str, torch.nn.Module] = {}
    rows: List[Dict[str, object]] = []
    for name, builder in builders.items():
        logger.info("Training model: %s", name)
        start = time.perf_counter()
        model = builder(input_size, model_config)
        trainer = Trainer(model, train_config, experiment_name=name)
        trainer.train(loaders["train"], loaders["val"], RMSELoss())
        best = trainer.load_best_model()
        elapsed = time.perf_counter() - start

        train_rmse = rmse(
            tensors["train"][1].numpy(),
            _torch_predict(best, tensors["train"][0]),
        )
        val_rmse = rmse(
            tensors["val"][1].numpy(),
            _torch_predict(best, tensors["val"][0]),
        )
        models[name] = best
        rows.append(
            {
                "model": name,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "time_s": elapsed,
            }
        )
        logger.info(
            "| %-12s | %10.4f | %10.4f | %7.1fs |",
            name,
            train_rmse,
            val_rmse,
            elapsed,
        )
    return models, rows


def _train_tree_models(
    tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Train the XGBoost and random-forest baselines on flattened windows.

    Args:
        tensors: Mapping of split name to ``(X, y)`` tensors.

    Returns:
        A tuple ``(models, rows)`` of trained models and progress rows.
    """
    config = ModelConfig()

    def _flatten(split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the last-timestep feature matrix and targets for a split.

        Args:
            split: Split name.

        Returns:
            A tuple ``(X_2d, y)``.
        """
        X, y = tensors[split]
        return X[:, -1, :].numpy(), y.numpy()

    X_train, y_train = _flatten("train")
    X_val, y_val = _flatten("val")

    models: Dict[str, object] = {}
    rows: List[Dict[str, object]] = []

    logger.info("Training model: xgboost")
    start = time.perf_counter()
    xgb = XGBoostRUL(config)
    xgb.train(X_train, y_train, X_val, y_val)
    elapsed = time.perf_counter() - start
    models["xgboost"] = xgb
    rows.append(
        {
            "model": "xgboost",
            "train_rmse": rmse(y_train, xgb.predict(X_train)),
            "val_rmse": rmse(y_val, xgb.predict(X_val)),
            "time_s": elapsed,
        }
    )
    logger.info(
        "| %-12s | %10.4f | %10.4f | %7.1fs |",
        "xgboost",
        rows[-1]["train_rmse"],
        rows[-1]["val_rmse"],
        elapsed,
    )

    logger.info("Training model: rf")
    start = time.perf_counter()
    forest = RandomForestRUL(config)
    forest.train(X_train, y_train)
    elapsed = time.perf_counter() - start
    models["rf"] = forest
    rows.append(
        {
            "model": "rf",
            "train_rmse": rmse(y_train, forest.predict(X_train)),
            "val_rmse": rmse(y_val, forest.predict(X_val)),
            "time_s": elapsed,
        }
    )
    logger.info(
        "| %-12s | %10.4f | %10.4f | %7.1fs |",
        "rf",
        rows[-1]["train_rmse"],
        rows[-1]["val_rmse"],
        elapsed,
    )
    return models, rows


def _log_table(title: str, rows: List[Dict[str, object]]) -> None:
    """Log a formatted progress or comparison table.

    Args:
        title: A heading for the table.
        rows: Per-model metric rows.
    """
    logger.info(title)
    logger.info("| %-12s | %-10s | %-10s | %-8s |", "Model", "Train RMSE", "Val RMSE", "Time")
    logger.info("|%s|%s|%s|%s|", "-" * 14, "-" * 12, "-" * 12, "-" * 10)
    for row in rows:
        logger.info(
            "| %-12s | %10.4f | %10.4f | %7.1fs |",
            row["model"],
            float(row["train_rmse"]),
            float(row["val_rmse"]),
            float(row["time_s"]),
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training workflow.

    Returns:
        The parsed arguments namespace with ``subset`` and ``epochs``.
    """
    parser = argparse.ArgumentParser(
        description="Train all SentinelAI RUL models sequentially."
    )
    parser.add_argument(
        "--subset",
        default=_SUBSET,
        help="C-MAPSS subset to train on (e.g. FD001).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainingConfig().max_epochs,
        help="Maximum training epochs per deep model.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full sequential training workflow."""
    global _SUBSET
    args = _parse_args()
    _SUBSET = args.subset
    set_seed(_SEED)
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Training on subset=%s for up to %d epochs", _SUBSET, args.epochs)

    loaders, tensors, feature_cols = _build_datasets()
    input_size = len(feature_cols)
    logger.info("Pipeline ready: input_size=%d", input_size)

    torch_models, torch_rows = _train_torch_models(
        loaders, tensors, input_size, args.epochs
    )
    tree_models, tree_rows = _train_tree_models(tensors)
    progress_rows = torch_rows + tree_rows
    _log_table("=== Training progress ===", progress_rows)

    # Test-set comparison of every trained model.
    X_test, y_test_tensor = tensors["test"]
    y_test = y_test_tensor.numpy()
    X_test_2d = X_test[:, -1, :].numpy()

    comparison: List[Dict[str, object]] = []
    test_predictions: Dict[str, np.ndarray] = {}
    for name, model in torch_models.items():
        preds = _torch_predict(model, X_test)
        test_predictions[name] = preds
        comparison.append({"model": name, **_metrics(y_test, preds)})
    for name, model in tree_models.items():
        preds = np.asarray(model.predict(X_test_2d)).reshape(-1)
        test_predictions[name] = preds
        comparison.append({"model": name, **_metrics(y_test, preds)})

    comparison_frame = pd.DataFrame(comparison).sort_values("rmse")
    logger.info("=== Final test-set comparison ===\n%s",
                comparison_frame.to_string(index=False))

    # Inverse-RMSE ensemble weights and production bundle.
    best_name = str(comparison_frame.iloc[0]["model"])
    rmse_by_model = {
        str(row["model"]): float(row["rmse"]) for row in comparison
    }
    inverse = {n: 1.0 / max(v, 1e-6) for n, v in rmse_by_model.items()}
    total = sum(inverse.values()) or 1.0
    ensemble_weights = {n: w / total for n, w in inverse.items()}

    checkpoints: Dict[str, str] = {}
    for name, model in torch_models.items():
        path = _CHECKPOINT_DIR / f"{name}_best.pt"
        checkpoints[name] = str(path)
    for name, model in tree_models.items():
        path = _CHECKPOINT_DIR / f"{name}.joblib"
        model.save(path)
        checkpoints[name] = str(path)

    # The inference service consumes a single deep model, so the best
    # torch model is always persisted as best_model.pt — even when a tree
    # model wins the overall comparison.
    best_torch_name = next(
        (
            str(row["model"])
            for _, row in comparison_frame.iterrows()
            if str(row["model"]) in torch_models
        ),
        None,
    )
    if best_torch_name is not None:
        save_model(
            torch_models[best_torch_name],
            _BEST_MODEL_PATH,
            metadata={"model": best_torch_name, "input_size": input_size},
        )
        logger.info(
            "Saved best deep model '%s' to %s",
            best_torch_name,
            _BEST_MODEL_PATH,
        )

    bundle = {
        "best_model": best_name,
        "input_size": input_size,
        "feature_columns": feature_cols,
        "checkpoints": checkpoints,
        "ensemble_weights": ensemble_weights,
        "metrics": {row["model"]: row for row in comparison},
        "window_size": _WINDOW_SIZE,
    }
    joblib.dump(bundle, _PRODUCTION_PATH)
    logger.info(
        "Saved production model bundle (best=%s) to %s",
        best_name,
        _PRODUCTION_PATH,
    )


if __name__ == "__main__":
    main()
