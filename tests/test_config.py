from pathlib import Path

import pytest
import yaml

from ecg_denoising.config import load_config


def test_quick_config_is_valid() -> None:
    config = load_config(Path("configs/quick.yaml"))
    assert config.data.samples == 500
    assert config.model.kernel_size % 2 == 1


def test_config_rejects_grouped_split_with_empty_partition(tmp_path) -> None:
    raw = yaml.safe_load(Path("configs/quick.yaml").read_text(encoding="utf-8"))
    raw["data"].update(
        {
            "subjects": 3,
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
        }
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="at least one subject"):
        load_config(path)
