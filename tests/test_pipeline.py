from pathlib import Path

from ecg_denoising.config import load_config
from ecg_denoising.data import prepare_dataset
from ecg_denoising.evaluate import evaluate_model
from ecg_denoising.train import train_model


def test_quick_pipeline(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path(__file__).parents[1] / "configs" / "quick.yaml"
    config = load_config(config_path)
    dataset = prepare_dataset(
        config,
        "data/processed/dataset.npz",
        "reports/data_manifest.json",
    )
    summary = train_model(config, dataset, "models/model.pt", "reports/training_summary.json")
    metrics = evaluate_model(config, dataset, "models/model.pt", "reports/metrics.json")
    assert summary["epochs_completed"] == 1
    assert Path("models/model.pt").exists()
    assert Path("reports/data_manifest.json").exists()
    assert Path("reports/predictions.npz").exists()
    assert Path("reports/promotion.json").exists()
    assert "snr_improvement_db" in metrics
