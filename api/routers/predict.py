"""Prediction endpoints: RUL, anomaly detection, batch and history."""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.database import Alert, Prediction, SensorReading, get_db
from api.security import verify_api_key
from api.schemas.request import (
    BatchPredictRequest,
    PredictAnomalyRequest,
    PredictRULRequest,
)
from api.schemas.response import (
    ConfidenceInterval,
    ExplanationResponse,
    FeatureContribution,
    PredictAnomalyResponse,
    PredictRULResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Every prediction endpoint is gated by the X-API-Key dependency. The
# check is a no-op when SENTINEL_API_KEY is unset (see api.security).
router = APIRouter(
    prefix="/api/v1/predict",
    tags=["predict"],
    dependencies=[Depends(verify_api_key)],
)

_HISTORY_LIMIT = 100


def _get_predictor(request: Request) -> Any:
    """Return the shared :class:`Predictor` or raise if unavailable.

    Args:
        request: The incoming request carrying application state.

    Returns:
        The loaded predictor instance.

    Raises:
        HTTPException: If the predictor has not been initialized.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded.")
    return predictor


def _build_response(machine_id: str, result: Dict[str, Any]) -> PredictRULResponse:
    """Convert a raw predictor result into a response model.

    Args:
        machine_id: Identifier of the predicted machine.
        result: The dict returned by ``Predictor.predict_single``.

    Returns:
        A populated :class:`PredictRULResponse`.
    """
    top = [
        FeatureContribution(**feature)
        for feature in result.get("top_features", [])
    ]
    explanation = ExplanationResponse(
        top_features=top,
        summary_text=result.get("explanation", ""),
        attention_heatmap=result.get("attention_heatmap"),
    )
    interval = result["confidence_interval"]
    return PredictRULResponse(
        machine_id=machine_id,
        predicted_rul=result["predicted_rul"],
        confidence_interval=ConfidenceInterval(
            lower=interval["lower"], upper=interval["upper"]
        ),
        risk_level=result["risk_level"],
        risk_score=result["risk_score"],
        explanation=explanation,
        model_contributions=result.get("model_contributions", {}),
        timestamp=result.get("timestamp", datetime.utcnow()),
    )


def _persist_prediction(
    db: Session, machine_id: str, result: Dict[str, Any]
) -> None:
    """Store a prediction and raise a CRITICAL alert on risk escalation.

    Args:
        db: Active database session.
        machine_id: Identifier of the predicted machine.
        result: The dict returned by ``Predictor.predict_single``.
    """
    previous = (
        db.query(Prediction)
        .filter(Prediction.machine_id == machine_id)
        .order_by(Prediction.timestamp.desc())
        .first()
    )
    interval = result["confidence_interval"]
    record = Prediction(
        machine_id=machine_id,
        timestamp=result.get("timestamp", datetime.utcnow()),
        rul=result["predicted_rul"],
        anomaly_score=result.get("anomaly_score", 0.0),
        confidence_lower=interval["lower"],
        confidence_upper=interval["upper"],
        risk_level=result["risk_level"],
        risk_score=result["risk_score"],
        top_features=json.dumps(result.get("top_features", [])),
        model_contributions=json.dumps(result.get("model_contributions", {})),
    )
    db.add(record)

    became_critical = result["risk_level"] == "CRITICAL" and (
        previous is None or previous.risk_level != "CRITICAL"
    )
    if became_critical:
        db.add(
            Alert(
                machine_id=machine_id,
                timestamp=datetime.utcnow(),
                alert_type="risk_escalation",
                severity="critical",
                message=(
                    f"Machine {machine_id} entered CRITICAL state with "
                    f"predicted RUL {result['predicted_rul']:.1f} cycles."
                ),
            )
        )
        logger.warning("Raised CRITICAL alert for machine %s", machine_id)
    db.commit()


@router.post("/rul", response_model=PredictRULResponse)
def predict_rul(
    payload: PredictRULRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictRULResponse:
    """Predict the remaining useful life of a single machine.

    Args:
        payload: The RUL prediction request.
        request: The incoming request carrying the predictor.
        db: Active database session.

    Returns:
        The RUL prediction response.

    Raises:
        HTTPException: On invalid input or prediction failure.
    """
    start = time.perf_counter()
    if not payload.sensor_readings:
        raise HTTPException(status_code=422, detail="No sensor readings.")

    predictor = _get_predictor(request)
    try:
        result = predictor.predict_single(
            payload.machine_id, payload.sensor_readings
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("RUL prediction failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Prediction failed."
        ) from exc

    _persist_prediction(db, payload.machine_id, result)
    elapsed = (time.perf_counter() - start) * 1000.0
    logger.info(
        "predict/rul machine=%s rul=%.2f %.1fms",
        payload.machine_id,
        result["predicted_rul"],
        elapsed,
    )
    return _build_response(payload.machine_id, result)


@router.post("/anomaly", response_model=PredictAnomalyResponse)
def predict_anomaly(
    payload: PredictAnomalyRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> PredictAnomalyResponse:
    """Detect sensor anomalies for a machine via the autoencoder.

    Args:
        payload: The anomaly-detection request.
        request: The incoming request carrying the predictor.
        db: Active database session.

    Returns:
        The anomaly-detection response.

    Raises:
        HTTPException: On invalid input or detection failure.
    """
    start = time.perf_counter()
    if not payload.sensor_readings:
        raise HTTPException(status_code=422, detail="No sensor readings.")

    predictor = _get_predictor(request)
    autoencoder = predictor._model_cache.get("autoencoder")
    if autoencoder is None:
        raise HTTPException(
            status_code=503, detail="Anomaly model not loaded."
        )

    try:
        tensor = predictor.preprocess(payload.sensor_readings)
        with torch.no_grad():
            reconstructed, _ = autoencoder(tensor)
            per_sensor = (
                ((tensor - reconstructed) ** 2)
                .mean(dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
            )
        is_anom, score = autoencoder.is_anomaly(tensor)
    except Exception as exc:  # noqa: BLE001
        logger.error("Anomaly detection failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Anomaly detection failed."
        ) from exc

    threshold = float(getattr(autoencoder, "threshold", 0.0))
    sensor_names = [f"feature_{i}" for i in range(len(per_sensor))]
    errors = {
        name: float(value)
        for name, value in zip(sensor_names, per_sensor)
    }
    cutoff = float(np.percentile(per_sensor, 90)) if per_sensor.size else 0.0
    anomalous = [name for name, e in errors.items() if e >= cutoff and is_anom]

    db.add(
        SensorReading(
            machine_id=payload.machine_id,
            timestamp=datetime.utcnow(),
            sensor_data=json.dumps(payload.sensor_readings),
        )
    )
    db.commit()

    elapsed = (time.perf_counter() - start) * 1000.0
    logger.info(
        "predict/anomaly machine=%s is_anomaly=%s %.1fms",
        payload.machine_id,
        is_anom,
        elapsed,
    )
    return PredictAnomalyResponse(
        is_anomaly=bool(is_anom),
        anomaly_score=float(score),
        threshold=threshold,
        anomalous_sensors=anomalous,
        reconstruction_error_per_sensor=errors,
    )


@router.post("/batch", response_model=List[PredictRULResponse])
async def predict_batch(
    payload: BatchPredictRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> List[PredictRULResponse]:
    """Predict RUL for many machines concurrently.

    Args:
        payload: The batch prediction request.
        request: The incoming request carrying the predictor.
        db: Active database session.

    Returns:
        A list of RUL prediction responses in request order.

    Raises:
        HTTPException: If the batch is empty.
    """
    start = time.perf_counter()
    if not payload.machines:
        raise HTTPException(status_code=422, detail="Empty batch.")

    predictor = _get_predictor(request)

    async def _one(item: PredictRULRequest) -> PredictRULResponse:
        """Run a single prediction off the event loop.

        Args:
            item: One machine's RUL request.

        Returns:
            The RUL prediction response for the machine.
        """
        result = await asyncio.to_thread(
            predictor.predict_single, item.machine_id, item.sensor_readings
        )
        _persist_prediction(db, item.machine_id, result)
        return _build_response(item.machine_id, result)

    responses = await asyncio.gather(
        *(_one(item) for item in payload.machines)
    )
    elapsed = (time.perf_counter() - start) * 1000.0
    logger.info(
        "predict/batch n=%d %.1fms", len(responses), elapsed
    )
    return list(responses)


@router.get("/history/{machine_id}")
def prediction_history(
    machine_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Return the recent prediction history with trend data for a machine.

    Args:
        machine_id: Identifier of the machine.
        db: Active database session.

    Returns:
        A dict with the prediction list and per-day RUL change trend.
    """
    records = (
        db.query(Prediction)
        .filter(Prediction.machine_id == machine_id)
        .order_by(Prediction.timestamp.desc())
        .limit(_HISTORY_LIMIT)
        .all()
    )
    ordered = list(reversed(records))
    history: List[Dict[str, Any]] = []
    for index, record in enumerate(ordered):
        rul_change = 0.0
        if index > 0:
            previous = ordered[index - 1]
            days = (
                record.timestamp - previous.timestamp
            ).total_seconds() / 86400.0
            if days > 0:
                rul_change = (record.rul - previous.rul) / days
        history.append(
            {
                "id": record.id,
                "timestamp": record.timestamp,
                "rul": record.rul,
                "risk_level": record.risk_level,
                "risk_score": record.risk_score,
                "anomaly_score": record.anomaly_score,
                "rul_change_per_day": round(rul_change, 4),
            }
        )
    logger.info(
        "predict/history machine=%s n=%d", machine_id, len(history)
    )
    return {"machine_id": machine_id, "count": len(history), "history": history}
