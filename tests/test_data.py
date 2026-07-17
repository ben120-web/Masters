import numpy as np

from ecg_denoising.config import load_config
from ecg_denoising.data import prepare_dataset


def test_subjects_do_not_cross_splits(tmp_path) -> None:
    output = prepare_dataset(load_config("configs/quick.yaml"), tmp_path / "dataset.npz")
    with np.load(output) as dataset:
        for subject in np.unique(dataset["subject_id"]):
            subject_splits = np.unique(dataset["split"][dataset["subject_id"] == subject])
            assert len(subject_splits) == 1
        assert set(np.unique(dataset["split"])) == {0, 1, 2}
