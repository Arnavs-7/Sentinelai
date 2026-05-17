"""Data and concept drift detection for deployed RUL models."""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import ks_2samp

from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42
_KS_ALPHA = 0.05
_PSI_MODERATE = 0.1
_PSI_SIGNIFICANT = 0.2


class DriftDetector:
    """Detects feature and concept drift against a reference distribution.

    Three complementary detectors are provided: a per-feature
    Kolmogorov-Smirnov test, the Population Stability Index, and a
    lightweight ADWIN-style change detector over prediction errors.
    """

    def __init__(
        self, reference_data: np.ndarray, feature_names: List[str]
    ) -> None:
        """Initialize the detector with reference data.

        Args:
            reference_data: 2-D array of shape ``(n_samples, n_features)``
                captured from the training distribution.
            feature_names: Names aligned to the feature axis.
        """
        self.reference_data = np.asarray(reference_data, dtype=float)
        self.feature_names = feature_names

    def ks_test(self, live_data: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run a Kolmogorov-Smirnov test per feature.

        Args:
            live_data: 2-D array of recent observations aligned to the
                reference feature axis.

        Returns:
            A mapping of feature name to a dict with the KS ``statistic``,
            ``p_value`` and a ``drifted`` flag.
        """
        live = np.asarray(live_data, dtype=float)
        results: Dict[str, Dict[str, Any]] = {}
        for index, name in enumerate(self.feature_names):
            statistic, p_value = ks_2samp(
                self.reference_data[:, index], live[:, index]
            )
            results[name] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drifted": bool(p_value < _KS_ALPHA),
            }
        n_drifted = sum(r["drifted"] for r in results.values())
        logger.info("KS test: %d/%d features drifted", n_drifted, len(results))
        return results

    def psi(
        self, live_data: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """Compute the Population Stability Index per feature.

        Bins are derived from reference-data quantiles; the PSI sums the
        weighted log-ratio of live versus reference bin proportions.

        Args:
            live_data: 2-D array of recent observations.
            n_bins: Number of quantile bins.

        Returns:
            A mapping of feature name to a dict with the ``psi_score`` and
            a ``severity`` label of ``none``, ``moderate`` or
            ``significant``.
        """
        live = np.asarray(live_data, dtype=float)
        eps = 1e-6
        results: Dict[str, Dict[str, Any]] = {}

        for index, name in enumerate(self.feature_names):
            reference = self.reference_data[:, index]
            quantiles = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.quantile(reference, quantiles)
            edges[0], edges[-1] = -np.inf, np.inf

            expected, _ = np.histogram(reference, bins=edges)
            actual, _ = np.histogram(live[:, index], bins=edges)
            expected_pct = expected / max(expected.sum(), 1) + eps
            actual_pct = actual / max(actual.sum(), 1) + eps

            psi_score = float(
                np.sum(
                    (actual_pct - expected_pct)
                    * np.log(actual_pct / expected_pct)
                )
            )
            if psi_score > _PSI_SIGNIFICANT:
                severity = "significant"
            elif psi_score > _PSI_MODERATE:
                severity = "moderate"
            else:
                severity = "none"
            results[name] = {"psi_score": psi_score, "severity": severity}

        logger.info("PSI computed for %d features", len(results))
        return results

    def detect_concept_drift(
        self, errors: List[float]
    ) -> Dict[str, Any]:
        """Detect concept drift via an ADWIN-style window comparison.

        The error stream is split at every candidate point; drift is
        flagged when the recent-window mean exceeds the historical-window
        mean by more than an adaptive threshold.

        Args:
            errors: Sequence of prediction errors ordered in time.

        Returns:
            A dict with a ``drift_detected`` flag and an optional
            ``change_point`` index.
        """
        series = np.asarray(errors, dtype=float)
        n = series.size
        if n < 4:
            return {"drift_detected": False, "change_point": None}

        overall_std = float(series.std()) or 1e-6
        change_point: Optional[int] = None

        for split in range(2, n - 1):
            historical = series[:split]
            recent = series[split:]
            delta = recent.mean() - historical.mean()
            threshold = overall_std * np.sqrt(
                0.5 * (1.0 / len(historical) + 1.0 / len(recent))
            ) * 3.0
            if delta > threshold:
                change_point = int(split)
                break

        drift_detected = change_point is not None
        logger.info(
            "Concept drift detected=%s change_point=%s",
            drift_detected,
            change_point,
        )
        return {
            "drift_detected": drift_detected,
            "change_point": change_point,
        }

    def full_report(
        self, live_data: np.ndarray, errors: List[float]
    ) -> Dict[str, Any]:
        """Run all detectors and assemble a combined drift report.

        Args:
            live_data: 2-D array of recent observations.
            errors: Sequence of recent prediction errors.

        Returns:
            A dict with the ``ks_test``, ``psi`` and ``concept_drift``
            results plus a ``summary`` of aggregate drift indicators.
        """
        ks_results = self.ks_test(live_data)
        psi_results = self.psi(live_data)
        concept = self.detect_concept_drift(errors)

        ks_drifted = [n for n, r in ks_results.items() if r["drifted"]]
        psi_significant = [
            n
            for n, r in psi_results.items()
            if r["severity"] == "significant"
        ]
        summary = {
            "n_features": len(self.feature_names),
            "ks_drifted_features": ks_drifted,
            "psi_significant_features": psi_significant,
            "concept_drift_detected": concept["drift_detected"],
            "drift_detected": bool(
                ks_drifted or psi_significant or concept["drift_detected"]
            ),
        }
        logger.info(
            "Drift report: overall drift_detected=%s",
            summary["drift_detected"],
        )
        return {
            "ks_test": ks_results,
            "psi": psi_results,
            "concept_drift": concept,
            "summary": summary,
        }
