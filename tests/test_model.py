import torch

from ecg_denoising.config import ModelConfig
from ecg_denoising.models import build_model


def test_model_preserves_shape_and_propagates_gradients() -> None:
    model = build_model(ModelConfig(channels=8, blocks=2, kernel_size=9))
    inputs = torch.randn(3, 1, 500, requires_grad=True)
    output = model(inputs)
    assert output.shape == inputs.shape
    output.square().mean().backward()
    assert model.encoder.weight.grad is not None
