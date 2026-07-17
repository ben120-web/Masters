from dataclasses import replace

import numpy as np
import pytest

from ecg_denoising.config import load_config
from ecg_denoising.data import DataContractError, load_split, prepare_dataset


def test_subjects_do_not_cross_splits(tmp_path) -> None:
    output = prepare_dataset(load_config("configs/quick.yaml"), tmp_path / "dataset.npz")
    with np.load(output) as dataset:
        for subject in np.unique(dataset["subject_id"]):
            subject_splits = np.unique(dataset["split"][dataset["subject_id"] == subject])
            assert len(subject_splits) == 1
        assert set(np.unique(dataset["split"])) == {0, 1, 2}


def test_dataset_contract_rejects_wrong_sampling_rate(tmp_path) -> None:
    config = load_config("configs/quick.yaml")
    output = prepare_dataset(config, tmp_path / "dataset.npz")
    incompatible = replace(config.data, sampling_rate_hz=config.data.sampling_rate_hz + 1)

    with pytest.raises(DataContractError, match="sampling rate"):
        load_split(output, 0, incompatible)
