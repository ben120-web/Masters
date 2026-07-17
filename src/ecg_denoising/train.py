"""Configuration-driven model training with MLflow and TensorBoard tracking."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .data import load_split
from .models import build_model
from .tracking import ExperimentTracker, TensorBoardTracker


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(path: Path, split: int, config: Config, shuffle: bool) -> DataLoader:
    noisy, clean, _ = load_split(path, split)
    dataset = TensorDataset(torch.from_numpy(noisy), torch.from_numpy(clean))
    generator = torch.Generator().manual_seed(config.project.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        generator=generator,
    )


def _mean_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for noisy, clean in loader:
            prediction = model(noisy.float().to(device))
            losses.append(nn.functional.mse_loss(prediction, clean.float().to(device)).item())
    return float(np.mean(losses))


def train_model(
    config: Config,
    dataset_path: str | Path = "data/processed/dataset.npz",
    model_path: str | Path = "models/model.pt",
    summary_path: str | Path = "reports/training_summary.json",
) -> dict[str, float | int | str]:
    random.seed(config.project.seed)
    np.random.seed(config.project.seed)
    torch.manual_seed(config.project.seed)
    device = resolve_device(config.training.device)
    dataset_path = Path(dataset_path)
    train_loader = _loader(dataset_path, 0, config, shuffle=True)
    validation_loader = _loader(dataset_path, 1, config, shuffle=False)
    model = build_model(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    tracker = ExperimentTracker(
        config.tracking.mlflow_tracking_uri, config.tracking.experiment_name
    )
    tensorboard = TensorBoardTracker(config.tracking.tensorboard_dir)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    epochs_completed = 0

    flat_params = {
        **{f"model.{key}": value for key, value in asdict(config.model).items()},
        **{f"training.{key}": value for key, value in asdict(config.training).items()},
        "seed": config.project.seed,
    }
    with tracker.run(run_name="train"):
        tracker.log_params(flat_params)
        for epoch in range(config.training.epochs):
            model.train()
            batch_losses: list[float] = []
            for noisy, clean in train_loader:
                noisy = noisy.float().to(device)
                clean = clean.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(noisy)
                loss = nn.functional.mse_loss(prediction, clean)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses))
            validation_loss = _mean_loss(model, validation_loader, device)
            epochs_completed = epoch + 1
            tensorboard.scalar("loss/train", train_loss, epoch)
            tensorboard.scalar("loss/validation", validation_loss, epoch)
            tracker.log_metrics(
                {"train_loss": train_loss, "validation_loss": validation_loss}, step=epoch
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= config.training.early_stopping_patience:
                    break

        tensorboard.close()
        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint")
        output = Path(model_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "model_config": asdict(config.model),
                "data_config": asdict(config.data),
                "seed": config.project.seed,
            },
            output,
        )
        summary: dict[str, float | int | str] = {
            "best_validation_loss": best_loss,
            "epochs_completed": epochs_completed,
            "device": str(device),
        }
        summary_output = Path(summary_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        tracker.log_artifact(output)
        tracker.log_artifact(summary_output)
    return summary
