"""Explicit, machine-readable candidate promotion decisions."""

from __future__ import annotations

from typing import Any

from .config import PromotionConfig


def assess_promotion(
    metrics: dict[str, float],
    by_snr: list[dict[str, float]],
    thresholds: PromotionConfig,
) -> dict[str, Any]:
    """Evaluate aggregate and worst-case SNR gates without hiding failures."""
    if not by_snr:
        raise ValueError("At least one per-SNR result is required for promotion")
    mean_improvement = float(metrics["snr_improvement_db"])
    worst_improvement = min(float(row["snr_improvement_db"]) for row in by_snr)
    checks = [
        {
            "name": "mean_snr_improvement_db",
            "value": mean_improvement,
            "operator": ">=",
            "threshold": thresholds.minimum_mean_snr_improvement_db,
            "passed": mean_improvement >= thresholds.minimum_mean_snr_improvement_db,
        },
        {
            "name": "worst_case_snr_improvement_db",
            "value": worst_improvement,
            "operator": ">=",
            "threshold": thresholds.minimum_worst_snr_improvement_db,
            "passed": worst_improvement >= thresholds.minimum_worst_snr_improvement_db,
        },
    ]
    approved = all(bool(check["passed"]) for check in checks)
    return {
        "schema_version": 1,
        "decision": "approved" if approved else "rejected",
        "checks": checks,
        "scope": "synthetic software-validation dataset only",
        "clinical_release_authorized": False,
    }
