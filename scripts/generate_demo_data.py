"""Generate realistic synthetic fleet data and populate the database.

Twelve turbofan engine units spanning three health states are
synthesized, each with 200 cycles of degrading sensor data. The result
is written to the SentinelAI SQLite database (engine units, sensor
readings, predictions and alerts) and to two processed CSV files so the
dashboard always has data to render on a fresh deployment.

The CSV files are written under ``/tmp/data`` because the application
directory is read-only on managed hosts such as Render.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.database import (
    Alert,
    Machine,
    Prediction,
    SensorReading,
    SessionLocal,
    init_db,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_N_CYCLES = 200
_N_PREDICTIONS = 30
# Written under /tmp because the app directory is read-only on Render.
_PROCESSED_DIR = Path("/tmp/data/processed")
_MACHINES_CSV = _PROCESSED_DIR / "demo_machines.csv"
_SENSORS_CSV = _PROCESSED_DIR / "demo_sensors.csv"

_SENSOR_IDS: List[int] = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
_SENSOR_COLUMNS: List[str] = [f"sensor_{i}" for i in _SENSOR_IDS]

# Approximate C-MAPSS FD001 baseline means and degradation direction.
_SENSOR_BASELINES: Dict[str, float] = {
    "sensor_2": 642.0,
    "sensor_3": 1590.0,
    "sensor_4": 1400.0,
    "sensor_7": 554.0,
    "sensor_8": 2388.0,
    "sensor_9": 9050.0,
    "sensor_11": 47.5,
    "sensor_12": 521.0,
    "sensor_13": 2388.0,
    "sensor_14": 8140.0,
    "sensor_15": 8.4,
    "sensor_17": 393.0,
    "sensor_20": 38.8,
    "sensor_21": 23.3,
}
_SENSOR_DIRECTION: Dict[str, int] = {
    "sensor_2": 1,
    "sensor_3": 1,
    "sensor_4": 1,
    "sensor_7": -1,
    "sensor_8": 1,
    "sensor_9": 1,
    "sensor_11": 1,
    "sensor_12": -1,
    "sensor_13": 1,
    "sensor_14": 1,
    "sensor_15": 1,
    "sensor_17": 1,
    "sensor_20": -1,
    "sensor_21": -1,
}

_LOCATIONS: List[str] = [
    "Hangar A - Bay 1",
    "Hangar A - Bay 2",
    "Hangar B - Bay 1",
    "Hangar B - Bay 2",
    "Maintenance Line C",
    "Maintenance Line D",
]


def _machine_specs() -> List[Dict[str, object]]:
    """Build the deterministic configuration for the twelve engine units.

    Four healthy, four warning and four critical turbofan units are
    produced with realistic remaining-useful-life ranges.

    Returns:
        A list of per-engine configuration dicts holding the identifier,
        health state, starting RUL and degradation rate.
    """
    rng = np.random.default_rng(_SEED)
    specs: List[Dict[str, object]] = []
    plan: List[Tuple[range, str, Tuple[float, float], float]] = [
        (range(1, 5), "HEALTHY", (120.0, 200.0), 0.1),
        (range(5, 9), "WARNING", (35.0, 80.0), 0.35),
        (range(9, 13), "CRITICAL", (5.0, 25.0), 0.6),
    ]
    for indices, state, rul_range, degradation in plan:
        for index in indices:
            low, high = rul_range
            specs.append(
                {
                    "id": f"ENGINE-{index:03d}",
                    "name": f"ENGINE-{index:03d}",
                    "type": "Turbofan",
                    "location": _LOCATIONS[(index - 1) % len(_LOCATIONS)],
                    "state": state,
                    "starting_rul": float(rng.uniform(low, high)),
                    "degradation_rate": degradation,
                    "status": "active",
                }
            )
    return specs


def _risk_from_rul(rul: float) -> Tuple[str, float]:
    """Map a RUL value to a categorical risk level and a risk score.

    Args:
        rul: Remaining useful life in cycles.

    Returns:
        A tuple ``(risk_level, risk_score)`` with the score in ``[0, 100]``.
    """
    if rul > 100.0:
        return "HEALTHY", float(max(0.0, 30.0 * (1.0 - (rul - 100.0) / 100.0)))
    if rul >= 30.0:
        return "WARNING", float(30.0 + 40.0 * (100.0 - rul) / 70.0)
    return "CRITICAL", float(70.0 + 30.0 * (1.0 - max(rul, 0.0) / 30.0))


def _generate_sensor_frame(
    machine_id: str, degradation_rate: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate 200 cycles of degrading sensor readings for one engine.

    Args:
        machine_id: Identifier of the engine unit.
        degradation_rate: Per-engine degradation severity.
        rng: Seeded random generator.

    Returns:
        A frame with ``machine_id``, ``cycle``, the 14 sensor columns and
        a derived ``rul`` column.
    """
    rows: List[Dict[str, float]] = []
    for cycle in range(1, _N_CYCLES + 1):
        progress = cycle / _N_CYCLES
        record: Dict[str, float] = {"machine_id": machine_id, "cycle": cycle}
        for column in _SENSOR_COLUMNS:
            mean = _SENSOR_BASELINES[column]
            healthy = float(rng.normal(mean, 0.01 * mean))
            noise = float(rng.normal(1.0, 0.03))
            factor = (
                degradation_rate
                * 0.05
                * (np.exp(progress) - 1.0)
                * noise
                * _SENSOR_DIRECTION[column]
            )
            record[column] = round(healthy * (1.0 + factor), 4)
        record["rul"] = round(max(0.0, _N_CYCLES - cycle), 2)
        rows.append(record)
    return pd.DataFrame(rows)


def _recent_timestamps(count: int) -> List[datetime]:
    """Return ``count`` timestamps evenly spaced over the last 24 hours.

    Args:
        count: Number of timestamps to produce.

    Returns:
        A list of timestamps ordered oldest-first, ending at the current
        UTC time.
    """
    now = datetime.utcnow()
    span = max(count - 1, 1)
    return [
        now - timedelta(hours=24.0 * (span - step) / span)
        for step in range(count)
    ]


def _prediction_trend(
    starting_rul: float, degradation_rate: float
) -> List[Tuple[datetime, float]]:
    """Build a descending RUL trend over the last 24 hours.

    Args:
        starting_rul: The engine's current remaining useful life.
        degradation_rate: Per-engine degradation severity.

    Returns:
        A list of ``(timestamp, rul)`` tuples ordered oldest-first; the
        final entry holds the engine's current RUL.
    """
    timestamps = _recent_timestamps(_N_PREDICTIONS)
    span = max(_N_PREDICTIONS - 1, 1)
    trend: List[Tuple[datetime, float]] = []
    for step, timestamp in enumerate(timestamps):
        steps_ago = span - step
        rul = starting_rul + degradation_rate * steps_ago * 0.6
        trend.append((timestamp, round(rul, 2)))
    return trend


def _top_features(rng: np.random.Generator) -> List[Dict[str, object]]:
    """Generate a plausible ranked top-feature attribution list.

    Args:
        rng: Seeded random generator.

    Returns:
        A list of five feature-contribution dicts.
    """
    chosen = rng.choice(_SENSOR_COLUMNS, size=5, replace=False)
    raw = np.sort(rng.uniform(0.05, 0.4, size=5))[::-1]
    total = float(raw.sum()) or 1.0
    features: List[Dict[str, object]] = []
    for name, weight in zip(chosen, raw):
        signed = float(rng.uniform(-1.0, 1.0))
        features.append(
            {
                "name": str(name),
                "contribution": float(weight / total),
                "direction": "up" if signed >= 0 else "down",
                "value": round(signed, 4),
            }
        )
    return features


def _reset_demo_rows(machine_ids: List[str]) -> None:
    """Delete any existing demo rows so the generator is idempotent.

    Args:
        machine_ids: Identifiers of the demo engine units to purge.
    """
    session = SessionLocal()
    try:
        for table in (Alert, Prediction, SensorReading):
            session.query(table).filter(
                table.machine_id.in_(machine_ids)
            ).delete(synchronize_session=False)
        session.query(Machine).filter(
            Machine.id.in_(machine_ids)
        ).delete(synchronize_session=False)
        session.commit()
        logger.info("Cleared previous demo rows for %d units", len(machine_ids))
    finally:
        session.close()


def generate() -> None:
    """Generate the synthetic fleet and persist it to the database and CSVs."""
    init_db()
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    specs = _machine_specs()
    machine_ids = [str(spec["id"]) for spec in specs]
    _reset_demo_rows(machine_ids)

    rng = np.random.default_rng(_SEED)
    install_base = datetime.utcnow() - timedelta(days=900)

    machine_records: List[Dict[str, object]] = []
    sensor_frames: List[pd.DataFrame] = []
    session = SessionLocal()
    try:
        for offset, spec in enumerate(specs):
            machine_id = str(spec["id"])
            degradation_rate = float(spec["degradation_rate"])
            install_date = install_base + timedelta(days=offset * 20)

            session.add(
                Machine(
                    id=machine_id,
                    name=str(spec["name"]),
                    type=str(spec["type"]),
                    location=str(spec["location"]),
                    install_date=install_date,
                    status=str(spec["status"]),
                    created_at=datetime.utcnow(),
                )
            )

            frame = _generate_sensor_frame(machine_id, degradation_rate, rng)
            sensor_frames.append(frame)

            reading_rows = list(frame.tail(_N_PREDICTIONS).iterrows())
            reading_times = _recent_timestamps(len(reading_rows))
            for (_, row), timestamp in zip(reading_rows, reading_times):
                session.add(
                    SensorReading(
                        machine_id=machine_id,
                        timestamp=timestamp,
                        sensor_data=json.dumps(
                            [float(row[c]) for c in _SENSOR_COLUMNS]
                        ),
                    )
                )

            trend = _prediction_trend(
                float(spec["starting_rul"]), degradation_rate
            )
            for timestamp, rul in trend:
                risk_level, risk_score = _risk_from_rul(rul)
                session.add(
                    Prediction(
                        machine_id=machine_id,
                        timestamp=timestamp,
                        rul=rul,
                        anomaly_score=round(
                            float(min(1.0, max(0.0, 1.0 - rul / 125.0))), 4
                        ),
                        confidence_lower=round(max(0.0, rul * 0.85), 2),
                        confidence_upper=round(rul * 1.15 + 2.0, 2),
                        risk_level=risk_level,
                        risk_score=round(risk_score, 2),
                        top_features=json.dumps(_top_features(rng)),
                        model_contributions=json.dumps(
                            {
                                "tcn": round(rul * 0.4, 3),
                                "transformer": round(rul * 0.35, 3),
                                "lstm": round(rul * 0.25, 3),
                            }
                        ),
                    )
                )

            current_rul = trend[-1][1]
            current_level, current_score = _risk_from_rul(current_rul)
            if current_level == "CRITICAL":
                session.add(
                    Alert(
                        machine_id=machine_id,
                        timestamp=datetime.utcnow(),
                        alert_type="risk_escalation",
                        severity="critical",
                        message=(
                            f"Engine {machine_id} is in a CRITICAL state "
                            f"with predicted RUL {current_rul:.1f} cycles; "
                            f"ground for inspection."
                        ),
                        acknowledged=False,
                    )
                )
            elif current_level == "WARNING":
                session.add(
                    Alert(
                        machine_id=machine_id,
                        timestamp=datetime.utcnow(),
                        alert_type="degradation",
                        severity="warning",
                        message=(
                            f"Engine {machine_id} shows accelerating "
                            f"degradation; schedule an inspection."
                        ),
                        acknowledged=False,
                    )
                )

            machine_records.append(
                {
                    "id": machine_id,
                    "name": str(spec["name"]),
                    "type": str(spec["type"]),
                    "location": str(spec["location"]),
                    "install_date": install_date.date().isoformat(),
                    "status": str(spec["status"]),
                    "state": str(spec["state"]),
                    "current_rul": current_rul,
                    "risk_level": current_level,
                    "risk_score": round(current_score, 2),
                    "degradation_rate": degradation_rate,
                }
            )
            logger.info(
                "Generated engine %s (%s, RUL=%.1f)",
                machine_id,
                current_level,
                current_rul,
            )

        session.commit()
    finally:
        session.close()

    pd.DataFrame(machine_records).to_csv(_MACHINES_CSV, index=False)
    sensors = pd.concat(sensor_frames, ignore_index=True)
    sensors.to_csv(_SENSORS_CSV, index=False)

    logger.info(
        "Demo data written: %d engine units, %d sensor rows -> %s, %s",
        len(machine_records),
        len(sensors),
        _MACHINES_CSV,
        _SENSORS_CSV,
    )


if __name__ == "__main__":
    generate()
