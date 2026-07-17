"""Validated checkpoint loading and framework-neutral denoising inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .config import DataConfig, ModelConfig
from .models import build_model
from .train import resolve_device


class CheckpointContractError(ValueError):
    """Raised when a checkpoint is incomplete or incompatible."""


class InputContractError(ValueError):
    """Raised when an inference request violates the model input contract."""


@dataclass(frozen=True)
class SignalContract:
    sampling_rate_hz: int
    segment_seconds: int

    @property
    def samples(self) -> int:
        return self.sampling_rate_hz * self.segment_seconds


class DenoisingPredictor:
    """Load one immutable model artifact and enforce its signal contract."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        contract: SignalContract,
        model_version: str,
    ) -> None:
        self._model = model
        self._device = device
        self.contract = contract
        self.model_version = model_version

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        requested_device: str = "auto",
    ) -> DenoisingPredictor:
        device = resolve_device(requested_device)
        raw: Any = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(raw, Mapping):
            raise CheckpointContractError("Checkpoint root must be a mapping")

        model_raw = raw.get("model_config")
        data_raw = raw.get("data_config")
        state_raw = raw.get("model_state")
        if not isinstance(model_raw, Mapping) or not isinstance(data_raw, Mapping):
            raise CheckpointContractError("Checkpoint must contain model_config and data_config")
        if not isinstance(state_raw, Mapping):
            raise CheckpointContractError("Checkpoint must contain a model_state mapping")

        try:
            model_config = ModelConfig(
                channels=int(model_raw["channels"]),
                blocks=int(model_raw["blocks"]),
                kernel_size=int(model_raw["kernel_size"]),
            )
            contract = SignalContract(
                sampling_rate_hz=int(data_raw["sampling_rate_hz"]),
                segment_seconds=int(data_raw["segment_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointContractError("Checkpoint configuration is invalid") from error
        if (
            model_config.channels < 1
            or model_config.blocks < 1
            or model_config.kernel_size < 1
            or model_config.kernel_size % 2 == 0
        ):
            raise CheckpointContractError("Checkpoint model dimensions are invalid")
        if contract.sampling_rate_hz < 1 or contract.segment_seconds < 1:
            raise CheckpointContractError("Checkpoint signal dimensions are invalid")

        model = build_model(model_config).to(device)
        state = cast(Mapping[str, torch.Tensor], state_raw)
        try:
            model.load_state_dict(state)
        except RuntimeError as error:
            raise CheckpointContractError(
                "Checkpoint weights do not match its model config"
            ) from error
        model.eval()
        return cls(model, device, contract, str(raw.get("package_version", "unknown")))

    def assert_data_config(self, config: DataConfig) -> None:
        if (
            config.sampling_rate_hz != self.contract.sampling_rate_hz
            or config.samples != self.contract.samples
        ):
            raise CheckpointContractError(
                "Checkpoint data contract does not match evaluation configuration: "
                f"expected {self.contract.sampling_rate_hz} Hz/{self.contract.samples} samples, "
                f"received {config.sampling_rate_hz} Hz/{config.samples} samples"
            )

    def denoise_batch(self, signals: np.ndarray, sampling_rate_hz: int) -> np.ndarray:
        try:
            values = np.asarray(signals, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise InputContractError("Signals must be a rectangular numeric array") from error
        if values.ndim != 3 or values.shape[1] != 1:
            raise InputContractError("Signals must have shape (batch, 1, samples)")
        if sampling_rate_hz != self.contract.sampling_rate_hz:
            raise InputContractError(f"sampling_rate_hz must be {self.contract.sampling_rate_hz}")
        if values.shape[-1] != self.contract.samples:
            raise InputContractError(f"Each signal must contain {self.contract.samples} samples")
        if values.shape[0] < 1 or not np.isfinite(values).all():
            raise InputContractError("Signals must be non-empty and contain only finite values")
        with torch.no_grad():
            tensor = torch.from_numpy(values).to(self._device)
            return self._model(tensor).cpu().numpy().astype(np.float32, copy=False)

    def denoise(self, signal: Sequence[float], sampling_rate_hz: int) -> np.ndarray:
        try:
            values = np.asarray(signal, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise InputContractError("Signal must be a numeric array") from error
        if values.ndim != 1:
            raise InputContractError("Signal must be one-dimensional")
        return self.denoise_batch(values[None, None, :], sampling_rate_hz)[0, 0]
