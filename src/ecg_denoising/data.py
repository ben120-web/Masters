"""Deterministic synthetic ECG preparation and grouped dataset loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config


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
    train_end = max(1, round(config.data.subjects * config.data.train_fraction))
    validation_end = max(
        train_end + 1,
        round(
            config.data.subjects * (config.data.train_fraction + config.data.validation_fraction)
        ),
    )
    validation_end = min(validation_end, config.data.subjects - 1)
    mapping: dict[int, int] = {}
    for subject in subject_ids[:train_end]:
        mapping[int(subject)] = 0
    for subject in subject_ids[train_end:validation_end]:
        mapping[int(subject)] = 1
    for subject in subject_ids[validation_end:]:
        mapping[int(subject)] = 2
    return mapping


def prepare_dataset(config: Config, output_path: str | Path) -> Path:
    """Create a deterministic smoke dataset while preventing subject leakage."""
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
    return output


def load_split(path: str | Path, split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as dataset:
        mask = dataset["split"] == split
        return dataset["noisy"][mask], dataset["clean"][mask], dataset["input_snr_db"][mask]
