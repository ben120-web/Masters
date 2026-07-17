"""Held-out evaluation and machine-readable reporting."""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .config import Config, ModelConfig
from .data import load_split
from .metrics import evaluate_metrics
from .models import build_model
from .train import resolve_device


def evaluate_model(
    config: Config,
    dataset_path: str | Path = "data/processed/dataset.npz",
    model_path: str | Path = "models/model.pt",
    report_path: str | Path = "reports/metrics.json",
) -> dict[str, float]:
    noisy, clean, input_snrs = load_split(dataset_path, 2)
    device = resolve_device(config.training.device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    allowed = {field.name for field in fields(ModelConfig)}
    model_config = ModelConfig(
        **{key: value for key, value in checkpoint["model_config"].items() if key in allowed}
    )
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(noisy).float().to(device)).cpu().numpy()

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
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_snr_db", "snr_improvement_db"])
        writer.writeheader()
        for snr_value in sorted(np.unique(input_snrs)):
            mask = input_snrs == snr_value
            row_metrics = evaluate_metrics(clean[mask], noisy[mask], predictions[mask])
            writer.writerow(
                {
                    "input_snr_db": float(snr_value),
                    "snr_improvement_db": row_metrics["snr_improvement_db"],
                }
            )
    return metrics
