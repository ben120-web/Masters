import numpy as np

from ecg_denoising.metrics import evaluate_metrics


def test_perfect_estimate_has_lower_error_and_better_snr() -> None:
    reference = np.sin(np.linspace(0, 4 * np.pi, 500, dtype=np.float32))
    noisy = reference + 0.2
    estimate = reference + 0.02
    metrics = evaluate_metrics(reference, noisy, estimate)
    assert metrics["rmse"] < 0.1
    assert metrics["normalized_correlation"] > 0.99
    assert metrics["snr_improvement_db"] > 0
