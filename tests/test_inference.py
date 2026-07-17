from dataclasses import replace

import numpy as np
import pytest

from ecg_denoising.config import load_config
from ecg_denoising.inference import (
    CheckpointContractError,
    DenoisingPredictor,
    InputContractError,
)


def test_predictor_enforces_signal_contract(model_checkpoint) -> None:
    config = load_config("configs/quick.yaml")
    predictor = DenoisingPredictor.from_checkpoint(model_checkpoint, "cpu")
    signal = np.zeros(config.data.samples, dtype=np.float32)

    prediction = predictor.denoise(signal, config.data.sampling_rate_hz)

    assert prediction.shape == signal.shape
    assert np.isfinite(prediction).all()
    with pytest.raises(InputContractError, match="sampling_rate_hz"):
        predictor.denoise(signal, config.data.sampling_rate_hz + 1)
    with pytest.raises(InputContractError, match="samples"):
        predictor.denoise(signal[:-1], config.data.sampling_rate_hz)


def test_predictor_rejects_evaluation_config_mismatch(model_checkpoint) -> None:
    config = load_config("configs/quick.yaml")
    predictor = DenoisingPredictor.from_checkpoint(model_checkpoint, "cpu")
    incompatible = replace(config.data, segment_seconds=config.data.segment_seconds + 1)

    with pytest.raises(CheckpointContractError, match="does not match"):
        predictor.assert_data_config(incompatible)
