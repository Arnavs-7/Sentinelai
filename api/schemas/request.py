"""Pydantic request models for the SentinelAI API."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictRULRequest(BaseModel):
    """Request body for a single remaining-useful-life prediction."""

    machine_id: str = Field(..., description="Identifier of the machine.")
    sensor_readings: List[List[float]] = Field(
        ..., description="Per-cycle rows of raw sensor readings."
    )
    window_size: int = Field(
        default=30, ge=1, description="Sliding-window length in cycles."
    )


class PredictAnomalyRequest(BaseModel):
    """Request body for a sensor anomaly-detection check."""

    machine_id: str = Field(..., description="Identifier of the machine.")
    sensor_readings: List[List[float]] = Field(
        ..., description="Per-cycle rows of raw sensor readings."
    )


class BatchPredictRequest(BaseModel):
    """Request body for a batch of RUL predictions."""

    machines: List[PredictRULRequest] = Field(
        ..., description="Per-machine RUL prediction requests."
    )


class CreateMachineRequest(BaseModel):
    """Request body for registering a new machine."""

    name: str = Field(..., description="Human-readable machine name.")
    type: str = Field(..., description="Machine type or model designation.")
    location: str = Field(..., description="Physical location of the machine.")
    install_date: datetime = Field(
        ..., description="Date the machine was installed."
    )


class UpdateMachineRequest(BaseModel):
    """Request body for partially updating a machine; all fields optional."""

    name: Optional[str] = Field(default=None, description="Updated name.")
    type: Optional[str] = Field(default=None, description="Updated type.")
    location: Optional[str] = Field(
        default=None, description="Updated location."
    )
    install_date: Optional[datetime] = Field(
        default=None, description="Updated install date."
    )
    status: Optional[str] = Field(
        default=None, description="Updated operational status."
    )
