"""Configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class DataConfig:
    sampling_rate_hz: int
    segment_seconds: int
    subjects: int
    segments_per_subject: int
    snr_db: tuple[float, ...]
    train_fraction: float
    validation_fraction: float

    @property
    def samples(self) -> int:
        return self.sampling_rate_hz * self.segment_seconds


@dataclass(frozen=True)
class ModelConfig:
    channels: int
    blocks: int
    kernel_size: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str
    early_stopping_patience: int


@dataclass(frozen=True)
class TrackingConfig:
    experiment_name: str
    mlflow_tracking_uri: str
    tensorboard_dir: str


@dataclass(frozen=True)
class PromotionConfig:
    minimum_mean_snr_improvement_db: float
    minimum_worst_snr_improvement_db: float


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    tracking: TrackingConfig
    promotion: PromotionConfig


def grouped_split_counts(data: DataConfig) -> tuple[int, int, int]:
    """Return deterministic subject counts for train, validation and test splits."""
    train_subjects = round(data.subjects * data.train_fraction)
    validation_subjects = round(data.subjects * data.validation_fraction)
    test_subjects = data.subjects - train_subjects - validation_subjects
    return train_subjects, validation_subjects, test_subjects


def _require(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{key}' must be a mapping")
    return value


def load_config(path: str | Path) -> Config:
    """Load a YAML configuration and enforce pipeline invariants."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    project = ProjectConfig(**_require(raw, "project"))
    data_raw = _require(raw, "data")
    data = DataConfig(**{**data_raw, "snr_db": tuple(float(x) for x in data_raw["snr_db"])})
    model = ModelConfig(**_require(raw, "model"))
    training = TrainingConfig(**_require(raw, "training"))
    tracking = TrackingConfig(**_require(raw, "tracking"))
    promotion = PromotionConfig(**_require(raw, "promotion"))

    if data.subjects < 3:
        raise ValueError("At least three subjects are required for grouped splits")
    if not 0 < data.train_fraction < 1 or not 0 <= data.validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if data.train_fraction + data.validation_fraction >= 1:
        raise ValueError("A non-empty test split is required")
    split_counts = grouped_split_counts(data)
    if any(count < 1 for count in split_counts):
        raise ValueError(
            "Grouped split fractions must allocate at least one subject to train, "
            "validation and test"
        )
    if data.sampling_rate_hz < 1 or data.segment_seconds < 1:
        raise ValueError("sampling_rate_hz and segment_seconds must be positive")
    if data.segments_per_subject < 1 or not data.snr_db:
        raise ValueError("segments_per_subject and snr_db must be non-empty")
    if model.channels < 1 or model.blocks < 1 or model.kernel_size < 1:
        raise ValueError("Model dimensions must be positive")
    if model.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd to preserve signal length")
    if not all(
        math.isfinite(value)
        for value in (
            promotion.minimum_mean_snr_improvement_db,
            promotion.minimum_worst_snr_improvement_db,
        )
    ):
        raise ValueError("Promotion thresholds must be finite")
    return Config(project, data, model, training, tracking, promotion)
