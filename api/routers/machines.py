"""Machine CRUD and dashboard endpoints for the SentinelAI API."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import Machine, Prediction, get_db
from api.schemas.request import CreateMachineRequest, UpdateMachineRequest
from api.schemas.response import (
    ConfidenceInterval,
    ExplanationResponse,
    FeatureContribution,
    MachineResponse,
    PredictRULResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/machines", tags=["machines"])

_DASHBOARD_LIMIT = 30


def _latest_prediction(
    db: Session, machine_id: str
) -> "PredictRULResponse | None":
    """Return the most recent stored prediction for a machine.

    Args:
        db: Active database session.
        machine_id: Identifier of the machine.

    Returns:
        The latest prediction as a response model, or ``None``.
    """
    record = (
        db.query(Prediction)
        .filter(Prediction.machine_id == machine_id)
        .order_by(Prediction.timestamp.desc())
        .first()
    )
    if record is None:
        return None
    return _prediction_to_response(record)


def _prediction_to_response(record: Prediction) -> PredictRULResponse:
    """Convert a stored prediction row into a response model.

    Args:
        record: The persisted prediction row.

    Returns:
        The corresponding :class:`PredictRULResponse`.
    """
    top = [
        FeatureContribution(**feature)
        for feature in json.loads(record.top_features or "[]")
    ]
    return PredictRULResponse(
        machine_id=record.machine_id,
        predicted_rul=record.rul,
        confidence_interval=ConfidenceInterval(
            lower=record.confidence_lower, upper=record.confidence_upper
        ),
        risk_level=record.risk_level,
        risk_score=record.risk_score,
        explanation=ExplanationResponse(top_features=top, summary_text=""),
        model_contributions=json.loads(record.model_contributions or "{}"),
        timestamp=record.timestamp,
    )


def _to_response(
    machine: Machine, db: Session, with_prediction: bool = True
) -> MachineResponse:
    """Build a :class:`MachineResponse` from a machine row.

    Args:
        machine: The persisted machine row.
        db: Active database session.
        with_prediction: Whether to attach the latest prediction.

    Returns:
        The machine response model.
    """
    return MachineResponse(
        id=machine.id,
        name=machine.name,
        type=machine.type,
        location=machine.location,
        install_date=machine.install_date,
        status=machine.status,
        created_at=machine.created_at,
        latest_prediction=(
            _latest_prediction(db, machine.id) if with_prediction else None
        ),
    )


def _get_or_404(db: Session, machine_id: str) -> Machine:
    """Fetch a machine by id or raise a 404 error.

    Args:
        db: Active database session.
        machine_id: Identifier of the machine.

    Returns:
        The machine row.

    Raises:
        HTTPException: If no machine with the id exists.
    """
    machine = db.query(Machine).filter(Machine.id == machine_id).first()
    if machine is None:
        raise HTTPException(status_code=404, detail="Machine not found.")
    return machine


@router.get("", response_model=List[MachineResponse])
def list_machines(db: Session = Depends(get_db)) -> List[MachineResponse]:
    """List all machines with their latest prediction.

    Args:
        db: Active database session.

    Returns:
        A list of machine responses.
    """
    machines = db.query(Machine).all()
    return [_to_response(machine, db) for machine in machines]


@router.post("", response_model=MachineResponse, status_code=201)
def create_machine(
    payload: CreateMachineRequest, db: Session = Depends(get_db)
) -> MachineResponse:
    """Register a new machine.

    Args:
        payload: The machine creation request.
        db: Active database session.

    Returns:
        The created machine response.
    """
    machine = Machine(
        id=str(uuid.uuid4()),
        name=payload.name,
        type=payload.type,
        location=payload.location,
        install_date=payload.install_date,
        status="active",
        created_at=datetime.utcnow(),
    )
    db.add(machine)
    db.commit()
    db.refresh(machine)
    logger.info("Created machine %s (%s)", machine.id, machine.name)
    return _to_response(machine, db, with_prediction=False)


@router.get("/{machine_id}", response_model=MachineResponse)
def get_machine(
    machine_id: str, db: Session = Depends(get_db)
) -> MachineResponse:
    """Retrieve a single machine by id.

    Args:
        machine_id: Identifier of the machine.
        db: Active database session.

    Returns:
        The machine response.
    """
    return _to_response(_get_or_404(db, machine_id), db)


@router.put("/{machine_id}", response_model=MachineResponse)
def update_machine(
    machine_id: str,
    payload: UpdateMachineRequest,
    db: Session = Depends(get_db),
) -> MachineResponse:
    """Apply a partial update to a machine.

    Args:
        machine_id: Identifier of the machine.
        payload: The fields to update; unset fields are ignored.
        db: Active database session.

    Returns:
        The updated machine response.
    """
    machine = _get_or_404(db, machine_id)
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(machine, field, value)
    db.commit()
    db.refresh(machine)
    logger.info("Updated machine %s fields=%s", machine_id, list(updates))
    return _to_response(machine, db)


@router.delete("/{machine_id}")
def delete_machine(
    machine_id: str, db: Session = Depends(get_db)
) -> Dict[str, bool]:
    """Delete a machine by id.

    Args:
        machine_id: Identifier of the machine.
        db: Active database session.

    Returns:
        A dict confirming deletion.
    """
    machine = _get_or_404(db, machine_id)
    db.delete(machine)
    db.commit()
    logger.info("Deleted machine %s", machine_id)
    return {"deleted": True}


@router.get("/{machine_id}/dashboard")
def machine_dashboard(
    machine_id: str, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Return full dashboard data for a machine.

    Args:
        machine_id: Identifier of the machine.
        db: Active database session.

    Returns:
        A dict with the machine, its latest prediction and the last
        30 predictions ordered chronologically.
    """
    machine = _get_or_404(db, machine_id)
    records = (
        db.query(Prediction)
        .filter(Prediction.machine_id == machine_id)
        .order_by(Prediction.timestamp.desc())
        .limit(_DASHBOARD_LIMIT)
        .all()
    )
    predictions = [
        _prediction_to_response(record) for record in reversed(records)
    ]
    return {
        "machine": _to_response(machine, db),
        "predictions": predictions,
        "prediction_count": len(predictions),
    }
