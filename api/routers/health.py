"""Health-check endpoints for the SentinelAI API."""

import os
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Request
from sqlalchemy import text

from api.database import engine
from src.utils.logger import get_logger
from src.utils.paths import MODEL_SEARCH_DIRS, resolve_model_path

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

_VERSION = "1.0.0"
_MODEL_KEYS = ("lstm", "tcn", "transformer", "autoencoder", "ensemble")


@router.get("/health")
def health(request: Request) -> Dict[str, Any]:
    """Return a basic liveness payload.

    Args:
        request: The incoming request, used to inspect loaded models.

    Returns:
        A dict with the service status, timestamp, version and a flag
        indicating whether prediction models are loaded.
    """
    predictor = getattr(request.app.state, "predictor", None)
    models_loaded = bool(
        predictor is not None and getattr(predictor, "_model_cache", {})
    )
    return {
        "status": "ok",
        "timestamp": datetime.utcnow(),
        "version": _VERSION,
        "models_loaded": models_loaded,
    }


@router.get("/api/v1/health/detailed")
def detailed_health(request: Request) -> Dict[str, Any]:
    """Return a detailed component health report.

    Args:
        request: The incoming request, used to inspect loaded models.

    Returns:
        A dict reporting API, database and per-model health status.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_status = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.error("Database health check failed: %s", exc)
        database_status = "error"

    predictor = getattr(request.app.state, "predictor", None)
    cache = getattr(predictor, "_model_cache", {}) if predictor else {}
    models = {
        key: "loaded" if key in cache or "primary" in cache else "not loaded"
        for key in _MODEL_KEYS
    }

    # Surface model-artifact discovery and any startup load failure so a
    # deployment can be diagnosed without access to its log stream.
    artifacts: Dict[str, Any] = {}
    for filename in ("best_model.pt", "preprocessor.joblib"):
        resolved = resolve_model_path(filename)
        artifacts[filename] = {
            "resolved_path": str(resolved),
            "exists": resolved.is_file(),
            "size_bytes": resolved.stat().st_size if resolved.is_file() else 0,
        }

    return {
        "api": "ok",
        "database": database_status,
        "models": models,
        "diagnostics": {
            "cwd": os.getcwd(),
            "search_dirs": [str(directory) for directory in MODEL_SEARCH_DIRS],
            "artifacts": artifacts,
            "model_load_error": getattr(
                request.app.state, "model_load_error", "unknown"
            ),
        },
    }
