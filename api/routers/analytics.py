"""Fleet analytics endpoints for the SentinelAI API."""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.database import Alert, Machine, Prediction, get_db
from api.schemas.response import (
    FleetHealthResponse,
    ModelMetricsResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

_REPORTS_DIR = Path("reports")
_TREND_DAYS = 30
_ALERT_LIMIT = 50


def _latest_per_machine(db: Session) -> Dict[str, Prediction]:
    """Return the most recent prediction for each machine.

    Args:
        db: Active database session.

    Returns:
        A mapping of machine id to its latest prediction row.
    """
    records = (
        db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    )
    latest: Dict[str, Prediction] = {}
    for record in records:
        latest.setdefault(record.machine_id, record)
    return latest


@router.get("/fleet-health", response_model=FleetHealthResponse)
def fleet_health(db: Session = Depends(get_db)) -> FleetHealthResponse:
    """Aggregate fleet health from the latest prediction per machine.

    Args:
        db: Active database session.

    Returns:
        The fleet health counts response.
    """
    total = db.query(Machine).count()
    latest = _latest_per_machine(db)
    counts = {"healthy": 0, "warning": 0, "critical": 0, "failed": 0}
    for record in latest.values():
        level = record.risk_level.lower()
        if level in counts:
            counts[level] += 1
    logger.info("fleet-health total=%d %s", total, counts)
    return FleetHealthResponse(
        total=total,
        healthy=counts["healthy"],
        warning=counts["warning"],
        critical=counts["critical"],
        failed=counts["failed"],
    )


@router.get("/rul-trend/{machine_id}")
def rul_trend(
    machine_id: str, db: Session = Depends(get_db)
) -> Dict[str, List[Any]]:
    """Return a daily RUL trend for a machine over the last 30 days.

    Args:
        machine_id: Identifier of the machine.
        db: Active database session.

    Returns:
        A dict with parallel ``dates``, ``rul_values`` and
        ``risk_levels`` lists ordered by day.
    """
    cutoff = datetime.utcnow() - timedelta(days=_TREND_DAYS)
    records = (
        db.query(Prediction)
        .filter(
            Prediction.machine_id == machine_id,
            Prediction.timestamp >= cutoff,
        )
        .order_by(Prediction.timestamp.asc())
        .all()
    )
    by_day: Dict[str, List[Prediction]] = defaultdict(list)
    for record in records:
        by_day[record.timestamp.date().isoformat()].append(record)

    dates: List[str] = []
    rul_values: List[float] = []
    risk_levels: List[str] = []
    for day in sorted(by_day):
        day_records = by_day[day]
        dates.append(day)
        rul_values.append(
            sum(r.rul for r in day_records) / len(day_records)
        )
        risk_levels.append(day_records[-1].risk_level)

    logger.info("rul-trend machine=%s days=%d", machine_id, len(dates))
    return {
        "dates": dates,
        "rul_values": rul_values,
        "risk_levels": risk_levels,
    }


@router.get("/model-performance", response_model=List[ModelMetricsResponse])
def model_performance() -> List[ModelMetricsResponse]:
    """Load the latest evaluation report and return per-model metrics.

    Returns:
        A list of model metrics responses; empty if no report exists.
    """
    reports = sorted(
        _REPORTS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        logger.info("model-performance: no report files found")
        return []

    try:
        payload = json.loads(reports[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to read evaluation report: %s", exc)
        return []

    rows = payload.values() if isinstance(payload, dict) else payload
    metrics: List[ModelMetricsResponse] = []
    for row in rows:
        metrics.append(
            ModelMetricsResponse(
                model_name=str(row.get("model", row.get("model_name", "?"))),
                rmse=float(row.get("rmse", 0.0)),
                mae=float(row.get("mae", 0.0)),
                r2=float(row.get("r2", 0.0)),
                score_s=float(row.get("score_s", 0.0)),
                acc_10=float(row.get("acc_10", 0.0)),
                acc_15=float(row.get("acc_15", 0.0)),
                inference_ms=float(row.get("inference_ms", 0.0)),
            )
        )
    logger.info(
        "model-performance: loaded %d models from %s",
        len(metrics),
        reports[0].name,
    )
    return metrics


@router.get("/alerts")
def recent_alerts(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return the most recent alerts with a severity breakdown.

    Args:
        db: Active database session.

    Returns:
        A dict with the alert list and a per-severity summary.
    """
    records = (
        db.query(Alert)
        .order_by(Alert.timestamp.desc())
        .limit(_ALERT_LIMIT)
        .all()
    )
    summary = {"critical": 0, "warning": 0, "info": 0}
    alerts: List[Dict[str, Any]] = []
    for record in records:
        severity = record.severity.lower()
        if severity in summary:
            summary[severity] += 1
        alerts.append(
            {
                "id": record.id,
                "machine_id": record.machine_id,
                "timestamp": record.timestamp,
                "alert_type": record.alert_type,
                "severity": record.severity,
                "message": record.message,
                "acknowledged": record.acknowledged,
            }
        )
    logger.info("alerts: returned %d (%s)", len(alerts), summary)
    return {"alerts": alerts, "summary": summary}
