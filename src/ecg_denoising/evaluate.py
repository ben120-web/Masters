"""Held-out evaluation and machine-readable reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import Config
from .data import load_split
from .governance import assess_promotion
from .inference import DenoisingPredictor
from .metrics import evaluate_metrics


def evaluate_model(
    config: Config,
    dataset_path: str | Path = "data/processed/dataset.npz",
    model_path: str | Path = "models/model.pt",
    report_path: str | Path = "reports/metrics.json",
) -> dict[str, float]:
    noisy, clean, input_snrs = load_split(dataset_path, 2, config.data)
    predictor = DenoisingPredictor.from_checkpoint(model_path, config.training.device)
    predictor.assert_data_config(config.data)
    predictions = predictor.denoise_batch(noisy, config.data.sampling_rate_hz)

    metrics = evaluate_metrics(clean, noisy, predictions)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        report.parent / "predictions.npz",
        noisy=noisy,
        clean=clean,
        denoised=predictions,
        input_snr_db=input_snrs,
    )
    csv_path = report.parent / "metrics.csv"
    by_snr: list[dict[str, float]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_snr_db", "snr_improvement_db"])
        writer.writeheader()
        for snr_value in sorted(np.unique(input_snrs)):
            mask = input_snrs == snr_value
            row_metrics = evaluate_metrics(clean[mask], noisy[mask], predictions[mask])
            row = {
                "input_snr_db": float(snr_value),
                "snr_improvement_db": row_metrics["snr_improvement_db"],
            }
            by_snr.append(row)
            writer.writerow(row)
    promotion = assess_promotion(metrics, by_snr, config.promotion)
    (report.parent / "promotion.json").write_text(
        json.dumps(promotion, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics
