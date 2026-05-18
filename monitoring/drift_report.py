"""Data and concept drift reporting for the SentinelAI RUL model.

This module wraps :class:`src.inference.drift_detector.DriftDetector` —
which already implements Kolmogorov-Smirnov, Population Stability Index
and an ADWIN-style concept-drift test — into a single report function
plus a CLI. It deliberately reuses the project's own detector rather
than adding the heavyweight ``evidently`` dependency, which would inflate
the Render and Streamlit Cloud build images and risk version conflicts.

Run as a script to write a JSON report::

    python -m monitoring.drift_report
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.inference.drift_detector import DriftDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REPORT_PATH = Path("reports/drift_report.json")
_SENSOR_NAMES: List[str] = [f"sensor_{i}" for i in range(1, 15)]


def generate_report(
    reference: np.ndarray,
    live: np.ndarray,
    feature_names: List[str],
    errors: List[float],
) -> Dict[str, Any]:
    """Build a full drift report comparing live data to a reference set.

    Args:
        reference: 2-D training-distribution array ``(n_samples, n_features)``.
        live: 2-D recent-observation array aligned to the same features.
        feature_names: Feature names aligned to the column axis.
        errors: Recent prediction errors, ordered in time.

    Returns:
        The drift report dict produced by :meth:`DriftDetector.full_report`.
    """
    detector = DriftDetector(reference, feature_names)
    report = detector.full_report(live, errors)
    logger.info(
        "Drift report built: drift_detected=%s",
        report["summary"]["drift_detected"],
    )
    return report


def demo_report(seed: int = 42) -> Dict[str, Any]:
    """Generate a representative drift report on synthetic sensor data.

    A reference distribution is sampled, then a live distribution is
    drawn with a deliberate shift applied to a subset of sensors so the
    report exercises every detector. Used by the dashboard Data Health
    page and the CLI when no real telemetry snapshot is available.

    Args:
        seed: Seed for the random generator, for reproducibility.

    Returns:
        A drift report dict for the synthetic fleet.
    """
    rng = np.random.default_rng(seed)
    n_features = len(_SENSOR_NAMES)
    reference = rng.normal(0.0, 1.0, size=(500, n_features))

    live = rng.normal(0.0, 1.0, size=(200, n_features))
    # Inject drift into three sensors: a mean shift and a variance change.
    live[:, 2] += 1.8
    live[:, 6] += 2.4
    live[:, 9] *= 2.5

    # Prediction errors that escalate partway through the window.
    errors = list(rng.normal(4.0, 1.0, size=12)) + list(
        rng.normal(11.0, 1.5, size=12)
    )
    return generate_report(reference, live, _SENSOR_NAMES, errors)


def save_report(report: Dict[str, Any], path: Path = _REPORT_PATH) -> Path:
    """Write a drift report to disk as JSON.

    Args:
        report: The drift report dict.
        path: Destination path; parent directories are created.

    Returns:
        The path the report was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Drift report written to %s", path)
    return path


def main() -> None:
    """Generate a demo drift report and print a console summary."""
    report = demo_report()
    save_report(report)
    summary = report["summary"]
    status = "DRIFT DETECTED" if summary["drift_detected"] else "STABLE"
    print(f"SentinelAI drift report: {status}")
    print(f"  Features monitored:   {summary['n_features']}")
    print(f"  KS-drifted features:  {summary['ks_drifted_features']}")
    print(f"  PSI-significant:      {summary['psi_significant_features']}")
    print(f"  Concept drift:        {summary['concept_drift_detected']}")
    print(f"  Report written to:    {_REPORT_PATH}")


if __name__ == "__main__":
    main()
