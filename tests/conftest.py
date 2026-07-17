from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from ecg_denoising import __version__
from ecg_denoising.config import load_config
from ecg_denoising.models import build_model


@pytest.fixture
def model_checkpoint(tmp_path: Path) -> Path:
    config = load_config("configs/quick.yaml")
    torch.manual_seed(config.project.seed)
    model = build_model(config.model)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(config.model),
            "data_config": asdict(config.data),
            "checkpoint_format_version": 1,
            "package_version": __version__,
            "seed": config.project.seed,
        },
        path,
    )
    return path
