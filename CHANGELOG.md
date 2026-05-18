# Changelog

All notable changes to SentinelAI are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-05-18

First production release of the SentinelAI Aviation Turbofan Engine
Predictive Maintenance Platform.

### Added

#### Machine Learning
- RUL (Remaining Useful Life) prediction trained on the NASA C-MAPSS
  turbofan degradation dataset.
- Model suite: LSTM with attention, Temporal Convolutional Network,
  Transformer, XGBoost and Random Forest, combined by an ensemble layer.
- Autoencoder-based sensor anomaly detection.
- SHAP explainability with per-prediction feature attributions.
- Monte Carlo dropout uncertainty estimation and confidence intervals.
- ONNX export for portable, accelerated inference.
- MLflow tracking of parameters, metrics and artifacts during training.

#### API
- FastAPI service with interactive OpenAPI documentation.
- RUL, anomaly, JSON-batch and CSV-upload-batch prediction endpoints.
- Machine CRUD endpoints and fleet analytics endpoints.
- `X-API-Key` authentication, enforced when `SENTINEL_API_KEY` is set.
- Per-request UUID tracing returned via the `X-Request-ID` header.
- Structured logging, rate limiting and a standardized error envelope.

#### Dashboard
- Streamlit multipage dashboard: Fleet Overview, Engine Detail,
  RUL Predictions, Model Performance and Upload Data.
- Data Health page surfacing KS / PSI / concept-drift metrics.
- Live Monitor page with simulated real-time telemetry, a critical-RUL
  alert banner and an optional Slack webhook.
- PDF maintenance report generation with recommended actions.

#### Monitoring & Quality
- `monitoring/drift_report.py` data and concept drift reporting.
- GitHub Actions CI pipeline: ruff lint and pytest with coverage.
- pytest suite covering the API, models, data pipeline, inference and
  monitoring modules.

#### Infrastructure
- Docker and Docker Compose for the API, dashboard and MLflow server.
- Render deployment blueprint with a `/health` health check.
- Automatic twelve-engine demo fleet seeding on first API startup.

[1.0.0]: https://github.com/Arnavs-7/Sentinelai/releases/tag/v1.0.0
