"""Pydantic response models for the SentinelAI API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """A single feature's contribution to a prediction."""

    name: str = Field(..., description="Feature or sensor name.")
    contribution: float = Field(
        ..., description="Normalized importance in ``[0, 1]``."
    )
    direction: str = Field(..., description="Either ``up`` or ``down``.")
    value: float = Field(..., description="Signed attribution value.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "sensor_4",
                "contribution": 0.31,
                "direction": "up",
                "value": 0.18,
            }
        }
    }


class ExplanationResponse(BaseModel):
    """Explainability payload accompanying a prediction."""

    top_features: List[FeatureContribution] = Field(
        ..., description="Ranked top contributing features."
    )
    summary_text: str = Field(
        ..., description="Rule-based natural-language explanation."
    )
    attention_heatmap: Optional[str] = Field(
        default=None, description="Base64-encoded PNG attention heatmap."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "top_features": [
                    {
                        "name": "sensor_4",
                        "contribution": 0.31,
                        "direction": "up",
                        "value": 0.18,
                    }
                ],
                "summary_text": "Sensor sensor_4 has increased 31% from "
                "baseline — primary driver of the failure prediction.",
                "attention_heatmap": None,
            }
        }
    }


class ConfidenceInterval(BaseModel):
    """Lower and upper bounds of a prediction confidence interval."""

    lower: float = Field(..., description="Lower bound of the interval.")
    upper: float = Field(..., description="Upper bound of the interval.")

    model_config = {
        "json_schema_extra": {"example": {"lower": 78.4, "upper": 102.1}}
    }


class PredictRULResponse(BaseModel):
    """Response body for a remaining-useful-life prediction."""

    machine_id: str = Field(..., description="Identifier of the machine.")
    predicted_rul: float = Field(
        ..., description="Predicted remaining useful life in cycles."
    )
    confidence_interval: ConfidenceInterval = Field(
        ..., description="Confidence interval around the prediction."
    )
    risk_level: str = Field(
        ..., description="One of HEALTHY, WARNING or CRITICAL."
    )
    risk_score: float = Field(
        ..., description="Normalized risk score in ``[0, 100]``."
    )
    explanation: ExplanationResponse = Field(
        ..., description="Explainability payload for the prediction."
    )
    model_contributions: Dict[str, float] = Field(
        ..., description="Per-model contribution to the final prediction."
    )
    timestamp: datetime = Field(
        ..., description="Time the prediction was produced."
    )


class PredictAnomalyResponse(BaseModel):
    """Response body for a sensor anomaly-detection check."""

    is_anomaly: bool = Field(
        ..., description="Whether the window is anomalous."
    )
    anomaly_score: float = Field(
        ..., description="Reconstruction-based anomaly score."
    )
    threshold: float = Field(
        ..., description="Calibrated anomaly decision threshold."
    )
    anomalous_sensors: List[str] = Field(
        ..., description="Names of sensors flagged as anomalous."
    )
    reconstruction_error_per_sensor: Dict[str, float] = Field(
        ..., description="Per-sensor mean reconstruction error."
    )


class MachineResponse(BaseModel):
    """Response body describing a machine."""

    id: str = Field(..., description="Machine UUID.")
    name: str = Field(..., description="Human-readable machine name.")
    type: str = Field(..., description="Machine type designation.")
    location: str = Field(..., description="Physical location.")
    install_date: datetime = Field(..., description="Installation date.")
    status: str = Field(..., description="Operational status.")
    created_at: datetime = Field(..., description="Record creation time.")
    latest_prediction: Optional[PredictRULResponse] = Field(
        default=None, description="Most recent stored prediction."
    )


class FleetHealthResponse(BaseModel):
    """Aggregate health counts across the machine fleet."""

    total: int = Field(..., description="Total number of machines.")
    healthy: int = Field(..., description="Machines in a healthy state.")
    warning: int = Field(..., description="Machines in a warning state.")
    critical: int = Field(..., description="Machines in a critical state.")
    failed: int = Field(..., description="Machines in a failed state.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 42,
                "healthy": 30,
                "warning": 8,
                "critical": 3,
                "failed": 1,
            }
        }
    }


class AlertResponse(BaseModel):
    """Response body describing an operational alert."""

    id: int = Field(..., description="Alert identifier.")
    machine_id: str = Field(..., description="Associated machine UUID.")
    timestamp: datetime = Field(..., description="Time the alert was raised.")
    alert_type: str = Field(..., description="Category of the alert.")
    severity: str = Field(..., description="Alert severity level.")
    message: str = Field(..., description="Human-readable alert message.")
    acknowledged: bool = Field(
        ..., description="Whether the alert has been acknowledged."
    )


class ModelMetricsResponse(BaseModel):
    """Evaluation metrics for a single trained model."""

    model_name: str = Field(..., description="Name of the evaluated model.")
    rmse: float = Field(..., description="Root mean squared error.")
    mae: float = Field(..., description="Mean absolute error.")
    r2: float = Field(..., description="Coefficient of determination.")
    score_s: float = Field(..., description="NASA asymmetric score.")
    acc_10: float = Field(
        ..., description="Accuracy within 10 cycles tolerance."
    )
    acc_15: float = Field(
        ..., description="Accuracy within 15 cycles tolerance."
    )
    inference_ms: float = Field(
        ..., description="Mean inference latency in milliseconds."
    )

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "model_name": "lstm",
                "rmse": 12.4,
                "mae": 9.1,
                "r2": 0.87,
                "score_s": 312.0,
                "acc_10": 0.71,
                "acc_15": 0.85,
                "inference_ms": 4.3,
            }
        },
    }


class ErrorResponse(BaseModel):
    """Standard error payload returned by the API."""

    detail: str = Field(..., description="Human-readable error description.")
    code: int = Field(..., description="HTTP status code.")
    timestamp: datetime = Field(..., description="Time the error occurred.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Machine not found.",
                "code": 404,
                "timestamp": "2026-05-17T12:00:00",
            }
        }
    }
