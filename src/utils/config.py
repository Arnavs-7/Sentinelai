"""Application configuration via Pydantic settings."""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    """Configuration for data ingestion, windowing and feature engineering."""

    model_config = SettingsConfigDict(env_prefix="DATA_", extra="ignore")

    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    splits_path: Path = Path("data/splits")
    window_size: int = 30
    stride: int = 1
    rul_clip: int = 125
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    sensors_to_drop: List[int] = [1, 5, 6, 10, 16, 18, 19]
    operating_condition_clusters: int = 6
    rolling_windows: List[int] = [5, 10, 20, 30]


class ModelConfig(BaseSettings):
    """Architecture hyperparameters for the model components."""

    model_config = SettingsConfigDict(env_prefix="MODEL_", extra="ignore")

    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    tcn_channels: List[int] = [64, 64, 128, 128, 256, 256]
    tcn_kernel_size: int = 3
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    transformer_dropout: float = 0.1
    ae_encoder_hidden: List[int] = [64, 32]


class TrainingConfig(BaseSettings):
    """Optimization and training-loop hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="TRAIN_", extra="ignore")

    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 15
    warmup_steps: int = 100
    seed: int = 42
    device: str = "auto"


class InferenceConfig(BaseSettings):
    """Configuration for inference and uncertainty estimation."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_", extra="ignore")

    model_path: Path = Path("models_saved/best_model.pt")
    confidence_level: float = 0.90
    mc_dropout_samples: int = 50
    anomaly_threshold_percentile: float = 95.0


class APIConfig(BaseSettings):
    """Configuration for the HTTP API service."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = "sqlite:////tmp/sentinel.db"
    log_level: str = "INFO"
