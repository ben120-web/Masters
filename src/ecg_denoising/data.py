"""Deterministic synthetic ECG preparation and grouped dataset loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import Config, DataConfig, grouped_split_counts


class DataContractError(ValueError):
    """Raised when a stored dataset violates the training/inference contract."""


def _gaussian(phase: np.ndarray, centre: float, width: float, amplitude: float) -> np.ndarray:
    distance = (phase - centre + 0.5) % 1.0 - 0.5
    return amplitude * np.exp(-0.5 * (distance / width) ** 2)


def synthesize_ecg(
    rng: np.random.Generator, samples: int, sampling_rate: int, heart_rate: float
) -> np.ndarray:
    """Generate a simple Lead-II-like waveform with beat-to-beat variation."""
    time = np.arange(samples, dtype=np.float32) / sampling_rate
    phase = (time * heart_rate / 60.0 + rng.uniform(0, 1)) % 1.0
    morphology = rng.normal(1.0, 0.06, size=5)
    signal = (
        _gaussian(phase, 0.18, 0.025, 0.12 * morphology[0])
        + _gaussian(phase, 0.37, 0.012, -0.15 * morphology[1])
        + _gaussian(phase, 0.40, 0.010, 1.00 * morphology[2])
        + _gaussian(phase, 0.43, 0.014, -0.25 * morphology[3])
        + _gaussian(phase, 0.68, 0.045, 0.30 * morphology[4])
    )
    return signal.astype(np.float32)


def motion_artifact(rng: np.random.Generator, samples: int, sampling_rate: int) -> np.ndarray:
    """Generate non-stationary low-frequency drift and transient electrode motion."""
    time = np.arange(samples, dtype=np.float32) / sampling_rate
    drift = 0.6 * np.sin(2 * np.pi * rng.uniform(0.15, 1.2) * time + rng.uniform(0, 6))
    innovations = rng.normal(0, 1, samples)
    coloured = np.empty(samples, dtype=np.float64)
    coloured[0] = innovations[0]
    coefficient = rng.uniform(0.90, 0.99)
    for index in range(1, samples):
        coloured[index] = coefficient * coloured[index - 1] + innovations[index]
    coloured /= np.std(coloured) + 1e-8
    transient = np.zeros(samples)
    for _ in range(int(rng.integers(1, 5))):
        centre = int(rng.integers(0, samples))
        width = float(rng.uniform(0.015, 0.12) * sampling_rate)
        transient += rng.normal(0, 1) * np.exp(-0.5 * ((np.arange(samples) - centre) / width) ** 2)
    return (drift + 0.35 * coloured + transient).astype(np.float32)


def add_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = float(np.mean(clean**2))
    noise_power = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return (clean + scale * noise).astype(np.float32)


def _subject_splits(config: Config, rng: np.random.Generator) -> dict[int, int]:
    subject_ids = rng.permutation(config.data.subjects)
    train_subjects, validation_subjects, _ = grouped_split_counts(config.data)
    train_end = train_subjects
    validation_end = train_subjects + validation_subjects
    mapping: dict[int, int] = {}
    for subject in subject_ids[:train_end]:
        mapping[int(subject)] = 0
    for subject in subject_ids[train_end:validation_end]:
        mapping[int(subject)] = 1
    for subject in subject_ids[validation_end:]:
        mapping[int(subject)] = 2
    return mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset(
    config: Config,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Path:
    """Create a deterministic smoke dataset and optional lineage manifest."""
    rng = np.random.default_rng(config.project.seed)
    split_by_subject = _subject_splits(config, rng)
    clean_rows: list[np.ndarray] = []
    noisy_rows: list[np.ndarray] = []
    subjects: list[int] = []
    splits: list[int] = []
    snrs: list[float] = []

    for subject in range(config.data.subjects):
        base_rate = float(rng.uniform(50, 120))
        for _ in range(config.data.segments_per_subject):
            clean = synthesize_ecg(
                rng, config.data.samples, config.data.sampling_rate_hz, base_rate + rng.normal(0, 2)
            )
            noise = motion_artifact(rng, config.data.samples, config.data.sampling_rate_hz)
            input_snr = float(rng.choice(config.data.snr_db))
            noisy = add_at_snr(clean, noise, input_snr)
            clean_rows.append(clean)
            noisy_rows.append(noisy)
            subjects.append(subject)
            splits.append(split_by_subject[subject])
            snrs.append(input_snr)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        clean=np.stack(clean_rows)[:, None, :],
        noisy=np.stack(noisy_rows)[:, None, :],
        subject_id=np.asarray(subjects, dtype=np.int16),
        split=np.asarray(splits, dtype=np.int8),
        input_snr_db=np.asarray(snrs, dtype=np.float32),
        sampling_rate_hz=np.asarray(config.data.sampling_rate_hz),
    )
    if manifest_path is not None:
        split_values = np.asarray(splits, dtype=np.int8)
        subject_values = np.asarray(subjects, dtype=np.int16)
        split_names = {0: "train", 1: "validation", 2: "test"}
        manifest = {
            "schema_version": 1,
            "dataset_sha256": _sha256(output),
            "generator": "ecg_denoising.data.prepare_dataset",
            "seed": config.project.seed,
            "records": len(clean_rows),
            "subjects": config.data.subjects,
            "split_records": {
                name: int(np.sum(split_values == split_id))
                for split_id, name in split_names.items()
            },
            "split_subjects": {
                name: int(np.unique(subject_values[split_values == split_id]).size)
                for split_id, name in split_names.items()
            },
            "signal": {
                "dtype": "float32",
                "channels": 1,
                "samples": config.data.samples,
                "sampling_rate_hz": config.data.sampling_rate_hz,
            },
            "input_snr_db": sorted(float(value) for value in config.data.snr_db),
        }
        manifest_output = Path(manifest_path)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def load_split(
    path: str | Path,
    split: int,
    expected: DataConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one split after validating the persisted dataset contract."""
    with np.load(path) as dataset:
        required = {
            "clean",
            "noisy",
            "subject_id",
            "split",
            "input_snr_db",
            "sampling_rate_hz",
        }
        missing = required.difference(dataset.files)
        if missing:
            raise DataContractError(f"Dataset is missing required arrays: {sorted(missing)}")

        noisy = dataset["noisy"]
        clean = dataset["clean"]
        subjects = dataset["subject_id"]
        splits = dataset["split"]
        input_snrs = dataset["input_snr_db"]
        if noisy.shape != clean.shape or noisy.ndim != 3 or noisy.shape[1] != 1:
            raise DataContractError("clean and noisy arrays must have shape (records, 1, samples)")
        records = noisy.shape[0]
        if any(values.shape != (records,) for values in (subjects, splits, input_snrs)):
            raise DataContractError("Dataset metadata arrays must have one value per record")
        if not np.isfinite(noisy).all() or not np.isfinite(clean).all():
            raise DataContractError("Signal arrays must contain only finite values")
        for subject in np.unique(subjects):
            if np.unique(splits[subjects == subject]).size != 1:
                raise DataContractError(f"Subject {subject!r} crosses dataset splits")

        sampling_rate_hz = int(dataset["sampling_rate_hz"].item())
        if expected is not None:
            if sampling_rate_hz != expected.sampling_rate_hz:
                raise DataContractError(
                    "Dataset sampling rate does not match configuration: "
                    f"{sampling_rate_hz} != {expected.sampling_rate_hz}"
                )
            if noisy.shape[-1] != expected.samples:
                raise DataContractError(
                    "Dataset segment length does not match configuration: "
                    f"{noisy.shape[-1]} != {expected.samples}"
                )

        mask = splits == split
        if not np.any(mask):
            raise DataContractError(f"Dataset split {split} is empty")
        return noisy[mask], clean[mask], input_snrs[mask]
