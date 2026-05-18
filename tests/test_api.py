"""Tests for the SentinelAI FastAPI service endpoints."""

import io
from datetime import datetime
from typing import Any, Dict, Iterator, List

import pytest
from fastapi.testclient import TestClient

from api.main import app


class StubPredictor:
    """A deterministic stand-in for the inference predictor.

    Returns a fixed prediction payload so the API can be tested without
    loading real model weights.
    """

    def predict_single(
        self, machine_id: str, sensor_readings: List[List[float]]
    ) -> Dict[str, Any]:
        """Return a fixed RUL prediction result.

        Args:
            machine_id: Identifier of the machine.
            sensor_readings: Per-cycle sensor reading rows.

        Returns:
            A prediction result dict.
        """
        return {
            "machine_id": machine_id,
            "predicted_rul": 87.5,
            "confidence_interval": {"lower": 74.0, "upper": 101.0},
            "risk_level": "WARNING",
            "risk_score": 45.0,
            "top_features": [
                {
                    "name": "sensor_4",
                    "contribution": 0.3,
                    "direction": "up",
                    "value": 0.12,
                }
            ],
            "explanation": "Stub explanation for testing.",
            "model_contributions": {"primary": 87.5},
            "anomaly_score": 0.3,
            "timestamp": datetime.utcnow(),
        }


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Yield a test client with a stubbed predictor.

    Yields:
        A :class:`TestClient` bound to the application.
    """
    with TestClient(app) as test_client:
        app.state.predictor = StubPredictor()
        yield test_client


def _valid_readings() -> List[List[float]]:
    """Build a valid 30-cycle by 14-sensor reading matrix.

    Returns:
        A nested list of sensor readings.
    """
    return [[float(s) for s in range(14)] for _ in range(30)]


def test_health_check(client: TestClient) -> None:
    """The health endpoint reports an ``ok`` status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rul_valid(client: TestClient) -> None:
    """A valid RUL request returns a 200 with a float prediction."""
    response = client.post(
        "/api/v1/predict/rul",
        json={
            "machine_id": "test-machine",
            "sensor_readings": _valid_readings(),
        },
    )
    assert response.status_code == 200
    assert isinstance(response.json()["predicted_rul"], float)


def test_predict_rul_invalid(client: TestClient) -> None:
    """A malformed RUL request body is rejected with a 422."""
    response = client.post(
        "/api/v1/predict/rul",
        json={"machine_id": "test-machine", "sensor_readings": [1.0, 2.0, 3.0]},
    )
    assert response.status_code == 422


def test_create_machine(client: TestClient) -> None:
    """Creating a machine returns a 201 with a UUID identifier."""
    response = client.post(
        "/api/v1/machines",
        json={
            "name": "Test Unit",
            "type": "Turbofan",
            "location": "Plant A",
            "install_date": "2024-01-01T00:00:00",
        },
    )
    assert response.status_code == 201
    import uuid

    uuid.UUID(response.json()["id"])


def test_get_machines(client: TestClient) -> None:
    """Listing machines returns a 200 with a list payload."""
    response = client.get("/api/v1/machines")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_fleet_health(client: TestClient) -> None:
    """Fleet health returns a 200 with all aggregate count fields."""
    response = client.get("/api/v1/analytics/fleet-health")
    assert response.status_code == 200
    payload = response.json()
    for key in ("total", "healthy", "warning", "critical"):
        assert key in payload


def test_prediction_history(client: TestClient) -> None:
    """Prediction history returns a 200 with a list of entries."""
    response = client.get("/api/v1/predict/history/test-machine")
    assert response.status_code == 200
    assert isinstance(response.json()["history"], list)


def test_predict_batch(client: TestClient) -> None:
    """A JSON batch request returns one response per machine."""
    response = client.post(
        "/api/v1/predict/batch",
        json={
            "machines": [
                {"machine_id": "engine-1", "sensor_readings": _valid_readings()},
                {"machine_id": "engine-2", "sensor_readings": _valid_readings()},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all(isinstance(item["predicted_rul"], float) for item in body)


def _sample_csv() -> bytes:
    """Build a two-engine sensor CSV in the C-MAPSS column layout.

    Returns:
        UTF-8 encoded CSV bytes ready for a multipart upload.
    """
    header = "unit_id,cycle," + ",".join(f"sensor_{i}" for i in range(1, 15))
    lines = [header]
    for unit in (1, 2):
        for cycle in range(1, 36):
            sensors = ",".join(str(float(s)) for s in range(14))
            lines.append(f"{unit},{cycle},{sensors}")
    return ("\n".join(lines)).encode("utf-8")


def test_predict_batch_csv(client: TestClient) -> None:
    """A CSV upload returns per-engine predictions and a fleet summary."""
    response = client.post(
        "/api/v1/predict/batch/csv",
        files={"file": ("fleet.csv", io.BytesIO(_sample_csv()), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["summary"]["n_engines"] == 2
    assert "mean_rul" in body["summary"]


def test_predict_batch_csv_missing_sensors(client: TestClient) -> None:
    """A CSV without sensor columns is rejected with a 422."""
    csv_bytes = b"unit_id,cycle\n1,1\n1,2\n"
    response = client.post(
        "/api/v1/predict/batch/csv",
        files={"file": ("bad.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 422


def test_api_key_rejects_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With auth enabled, a request without X-API-Key is rejected with 401."""
    monkeypatch.setenv("SENTINEL_API_KEY", "top-secret")
    response = client.post(
        "/api/v1/predict/rul",
        json={"machine_id": "m", "sensor_readings": _valid_readings()},
    )
    assert response.status_code == 401


def test_api_key_rejects_wrong(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With auth enabled, a wrong X-API-Key is rejected with 401."""
    monkeypatch.setenv("SENTINEL_API_KEY", "top-secret")
    response = client.post(
        "/api/v1/predict/rul",
        json={"machine_id": "m", "sensor_readings": _valid_readings()},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_api_key_accepts_valid(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With auth enabled, the correct X-API-Key is accepted with 200."""
    monkeypatch.setenv("SENTINEL_API_KEY", "top-secret")
    response = client.post(
        "/api/v1/predict/rul",
        json={"machine_id": "m", "sensor_readings": _valid_readings()},
        headers={"X-API-Key": "top-secret"},
    )
    assert response.status_code == 200


def test_request_id_header(client: TestClient) -> None:
    """Every response carries an X-Request-ID correlation header."""
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")
