import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_denoising.train import _clone_state_dict, _mean_loss


def test_mean_loss_is_weighted_by_elements_not_batches() -> None:
    inputs = torch.zeros(3, 1)
    targets = torch.tensor([[0.0], [0.0], [3.0]])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    loss = _mean_loss(nn.Identity(), loader, torch.device("cpu"))

    assert loss == pytest.approx(3.0)


def test_checkpoint_snapshot_does_not_share_model_storage() -> None:
    model = nn.Linear(2, 1, bias=False)
    snapshot = _clone_state_dict(model)
    original = snapshot["weight"].clone()

    with torch.no_grad():
        model.weight.add_(10)

    assert torch.equal(snapshot["weight"], original)
    assert not torch.equal(snapshot["weight"], model.weight)
