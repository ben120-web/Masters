"""Maintained denoising architectures."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class ResidualDenoiser1D(nn.Module):
    """Predict motion artefact and subtract it from the observed ECG."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2
        self.encoder = nn.Conv1d(1, config.channels, config.kernel_size, padding=padding)
        self.blocks = nn.Sequential(
            *(ResidualBlock(config.channels, config.kernel_size) for _ in range(config.blocks))
        )
        self.noise_head = nn.Conv1d(config.channels, 1, config.kernel_size, padding=padding)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.gelu(self.encoder(noisy))
        estimated_noise = self.noise_head(self.blocks(features))
        return noisy - estimated_noise


def build_model(config: ModelConfig) -> ResidualDenoiser1D:
    return ResidualDenoiser1D(config)
