"""HTTP client for communicating with the SentinelAI FastAPI service."""

import os
from typing import Any, Dict, List

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

_TIMEOUT = 30.0


class SentinelAPIClient:
    """Thin wrapper around the SentinelAI REST API.

    Every method issues a single HTTP request, parses the JSON body and
    returns it. Network and decoding failures are logged and converted
    into empty payloads so the dashboard never crashes on a bad call.
    """

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the client.

        Args:
            base_url: Root URL of the SentinelAI API, without a path. When
                omitted, the ``API_URL`` environment variable is used,
                falling back to ``http://localhost:8000``.
        """
        resolved = base_url or os.getenv("API_URL", "http://localhost:8000")
        self.base_url = resolved.rstrip("/")
        self._session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        default: Any | None = None,
    ) -> Any:
        """Perform an HTTP request and return the parsed JSON body.

        Args:
            method: HTTP verb, e.g. ``"GET"`` or ``"POST"``.
            path: API path beginning with a slash.
            json: Optional JSON-serializable request body.
            default: Value returned when the request fails.

        Returns:
            The parsed response body, or ``default`` on any error.
        """
        url = f"{self.base_url}{path}"
        try:
            response = self._session.request(
                method, url, json=json, timeout=_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.error("API %s %s failed: %s", method, path, exc)
            return default
        except ValueError as exc:
            logger.error("API %s %s returned invalid JSON: %s", method, path, exc)
            return default

    def get_fleet_health(self) -> Dict[str, Any]:
        """Return aggregate fleet health counts.

        Returns:
            A dict with ``total``/``healthy``/``warning``/``critical``/
            ``failed`` counts, or an empty dict on failure.
        """
        return self._request(
            "GET", "/api/v1/analytics/fleet-health", default={}
        )

    def get_machines(self) -> List[Dict[str, Any]]:
        """Return all registered machines with their latest prediction.

        Returns:
            A list of machine dicts, or an empty list on failure.
        """
        return self._request("GET", "/api/v1/machines", default=[])

    def get_machine(self, machine_id: str) -> Dict[str, Any]:
        """Return a single machine by id.

        Args:
            machine_id: Identifier of the machine.

        Returns:
            The machine dict, or an empty dict on failure.
        """
        return self._request(
            "GET", f"/api/v1/machines/{machine_id}", default={}
        )

    def predict_rul(
        self, machine_id: str, sensor_readings: List[Any]
    ) -> Dict[str, Any]:
        """Request a remaining-useful-life prediction.

        Args:
            machine_id: Identifier of the machine.
            sensor_readings: Window of sensor rows.

        Returns:
            The prediction response dict, or an empty dict on failure.
        """
        return self._request(
            "POST",
            "/api/v1/predict/rul",
            json={"machine_id": machine_id, "sensor_readings": sensor_readings},
            default={},
        )

    def predict_anomaly(
        self, machine_id: str, sensor_readings: List[Any]
    ) -> Dict[str, Any]:
        """Request an anomaly-detection result.

        Args:
            machine_id: Identifier of the machine.
            sensor_readings: Window of sensor rows.

        Returns:
            The anomaly response dict, or an empty dict on failure.
        """
        return self._request(
            "POST",
            "/api/v1/predict/anomaly",
            json={"machine_id": machine_id, "sensor_readings": sensor_readings},
            default={},
        )

    def get_prediction_history(self, machine_id: str) -> List[Dict[str, Any]]:
        """Return the recent prediction history for a machine.

        Args:
            machine_id: Identifier of the machine.

        Returns:
            A list of history entries, or an empty list on failure.
        """
        payload = self._request(
            "GET", f"/api/v1/predict/history/{machine_id}", default={}
        )
        return payload.get("history", []) if isinstance(payload, dict) else []

    def get_rul_trend(self, machine_id: str) -> Dict[str, Any]:
        """Return the daily RUL trend for a machine.

        Args:
            machine_id: Identifier of the machine.

        Returns:
            A dict with ``dates``/``rul_values``/``risk_levels`` lists.
        """
        return self._request(
            "GET", f"/api/v1/analytics/rul-trend/{machine_id}", default={}
        )

    def get_model_performance(self) -> List[Dict[str, Any]]:
        """Return per-model evaluation metrics.

        Returns:
            A list of model metric dicts, or an empty list on failure.
        """
        return self._request(
            "GET", "/api/v1/analytics/model-performance", default=[]
        )

    def get_alerts(self) -> Dict[str, Any]:
        """Return recent alerts with a severity breakdown.

        Returns:
            A dict with ``alerts`` and ``summary`` keys.
        """
        return self._request("GET", "/api/v1/analytics/alerts", default={})

    def create_machine(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new machine.

        Args:
            data: The machine creation payload.

        Returns:
            The created machine dict, or an empty dict on failure.
        """
        return self._request(
            "POST", "/api/v1/machines", json=data, default={}
        )
