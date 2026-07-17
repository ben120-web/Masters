"""Scientific denoising metrics with numerical guardrails."""

from __future__ import annotations

import numpy as np


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def normalized_correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference_flat = reference.reshape(-1).astype(np.float64)
    estimate_flat = estimate.reshape(-1).astype(np.float64)
    reference_flat -= reference_flat.mean()
    estimate_flat -= estimate_flat.mean()
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(estimate_flat)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(reference_flat, estimate_flat) / denominator)


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    signal_power = float(np.mean(reference.astype(np.float64) ** 2))
    error_power = float(np.mean((reference.astype(np.float64) - estimate) ** 2))
    return float(10 * np.log10((signal_power + 1e-12) / (error_power + 1e-12)))


def evaluate_metrics(
    reference: np.ndarray, noisy: np.ndarray, estimate: np.ndarray
) -> dict[str, float]:
    before = snr_db(reference, noisy)
    after = snr_db(reference, estimate)
    return {
        "rmse": rmse(reference, estimate),
        "normalized_correlation": normalized_correlation(reference, estimate),
        "input_snr_db": before,
        "output_snr_db": after,
        "snr_improvement_db": after - before,
    }
