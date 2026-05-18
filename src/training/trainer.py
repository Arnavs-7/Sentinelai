"""Training loop with mixed precision, scheduling and MLflow tracking."""

from typing import Dict, List

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    WarmupCosineScheduler,
)
from src.utils.config import TrainingConfig
from src.utils.logger import get_logger
from src.models.model_utils import get_device, load_model, set_seed

logger = get_logger(__name__)

_CHECKPOINT_DIR = "models_saved"


class Trainer:
    """Supervised trainer for PyTorch RUL models.

    Wraps a full training loop with automatic mixed precision, gradient
    clipping, warmup-cosine scheduling, early stopping, best-model
    checkpointing and per-epoch MLflow logging.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        experiment_name: str,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The model to train.
            config: Optimization and training-loop hyperparameters.
            experiment_name: MLflow experiment and run identifier; also
                used to name the checkpoint file.
        """
        set_seed(config.seed)
        self.config = config
        self.experiment_name = experiment_name
        self.device = get_device()
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.checkpoint_path = (
            f"{_CHECKPOINT_DIR}/{experiment_name}_best.pt"
        )
        self.checkpoint = ModelCheckpoint(
            save_path=self.checkpoint_path, monitor="val_loss", mode="min"
        )
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience, mode="min"
        )
        self._use_amp = self.device.type == "cuda"

    def _forward_prediction(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Run the model and normalize its output to a prediction tensor.

        Args:
            batch_x: Input feature batch.

        Returns:
            The model's prediction tensor.
        """
        output = self.model(batch_x)
        if isinstance(output, tuple):
            return output[0]
        return output

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            train_loader: Loader over the training split.
            val_loader: Loader over the validation split.
            loss_fn: The loss module optimized during training.

        Returns:
            A history dict with ``train_loss``, ``val_loss`` and
            ``learning_rate`` lists indexed by epoch.
        """
        total_steps = self.config.max_epochs * max(len(train_loader), 1)
        scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.experiment_name):
            mlflow.log_params(self.config.model_dump())

            for epoch in range(1, self.config.max_epochs + 1):
                train_loss = self._run_train_epoch(
                    train_loader, loss_fn, scaler, scheduler, epoch
                )
                val_metrics = self.evaluate(val_loader, loss_fn)
                val_loss = val_metrics["loss"]

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["learning_rate"].append(scheduler.get_lr())

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_rmse": val_metrics["rmse"],
                        "val_mae": val_metrics["mae"],
                        "learning_rate": scheduler.get_lr(),
                    },
                    step=epoch,
                )

                self.checkpoint(val_loss, self.model, epoch)
                if self.early_stopping(val_loss):
                    logger.info("Stopping training early at epoch %d", epoch)
                    break

            mlflow.log_artifact(self.checkpoint_path)

        logger.info(
            "Training complete; best val_loss=%.6f at epoch %d",
            self.checkpoint.best_score,
            self.checkpoint.best_epoch,
        )
        return history

    def _run_train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        scaler: "torch.cuda.amp.GradScaler",
        scheduler: WarmupCosineScheduler,
        epoch: int,
    ) -> float:
        """Run a single training epoch.

        Args:
            train_loader: Loader over the training split.
            loss_fn: The loss module optimized during training.
            scaler: Gradient scaler for mixed-precision training.
            scheduler: The learning-rate scheduler stepped per batch.
            epoch: The current epoch index.

        Returns:
            The mean training loss over the epoch.
        """
        self.model.train()
        running_loss = 0.0
        batches = 0
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch}", leave=False
        )

        for batch_x, batch_y in progress:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self._use_amp):
                pred = self._forward_prediction(batch_x)
                loss = loss_fn(pred, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.grad_clip
            )
            scaler.step(self.optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            batches += 1
            progress.set_postfix(loss=running_loss / batches)

        return running_loss / max(batches, 1)

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, loss_fn: nn.Module
    ) -> Dict[str, float]:
        """Evaluate the model on a data split.

        Args:
            loader: Loader over the split to evaluate.
            loss_fn: The loss module used to compute the ``loss`` metric.

        Returns:
            A dict with ``loss``, ``rmse`` and ``mae`` metrics.
        """
        self.model.eval()
        total_loss = 0.0
        batches = 0
        preds: List[np.ndarray] = []
        targets: List[np.ndarray] = []

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            pred = self._forward_prediction(batch_x)
            total_loss += float(loss_fn(pred, batch_y).item())
            batches += 1
            preds.append(pred.detach().cpu().numpy().reshape(-1))
            targets.append(batch_y.detach().cpu().numpy().reshape(-1))

        pred_array = np.concatenate(preds) if preds else np.array([])
        target_array = np.concatenate(targets) if targets else np.array([])
        errors = pred_array - target_array
        return {
            "loss": total_loss / max(batches, 1),
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else 0.0,
            "mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        }

    def load_best_model(self) -> nn.Module:
        """Reload the best checkpoint saved during training.

        Returns:
            The model populated with the best-epoch weights.
        """
        self.model, metadata = load_model(self.model, self.checkpoint_path)
        self.model = self.model.to(self.device)
        logger.info("Loaded best model from epoch %s", metadata.get("epoch"))
        return self.model
