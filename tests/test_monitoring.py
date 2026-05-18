"""Tests for the drift monitoring report module."""

import numpy as np

from monitoring.drift_report import demo_report, generate_report


def test_demo_report_structure() -> None:
    """The demo report exposes every detector section and a summary."""
    report = demo_report()
    for key in ("ks_test", "psi", "concept_drift", "summary"):
        assert key in report
    assert report["summary"]["n_features"] == 14


def test_demo_report_detects_injected_drift() -> None:
    """The demo report flags drift, as the demo data injects a shift."""
    report = demo_report()
    assert report["summary"]["drift_detected"] is True
    assert report["summary"]["ks_drifted_features"]


def test_generate_report_no_drift() -> None:
    """Identical reference and live data raise no drift."""
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 1.0, size=(300, 4))
    names = [f"sensor_{i}" for i in range(4)]
    report = generate_report(reference, reference, names, [1.0] * 10)
    assert report["summary"]["drift_detected"] is False
