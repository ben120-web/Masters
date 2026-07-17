from pathlib import Path

from ecg_denoising.config import load_config


def test_quick_config_is_valid() -> None:
    config = load_config(Path("configs/quick.yaml"))
    assert config.data.samples == 500
    assert config.model.kernel_size % 2 == 1
