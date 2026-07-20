# MSc Engineering Portfolio

[![CI](https://github.com/ben120-web/Masters/actions/workflows/ci.yml/badge.svg)](https://github.com/ben120-web/Masters/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![DVC](https://img.shields.io/badge/data-DVC-945DD6)](dvc.yaml)
[![MLflow](https://img.shields.io/badge/experiments-MLflow-0194E2)](MLproject)

An engineering portfolio spanning biomedical machine learning, model-predictive
control, state estimation, signal processing and intelligent systems. The MSc
dissertation is developed as a production-oriented ECG motion-artefact
denoising system; the taught modules retain their original analysis and expose
clear, reproducible demonstrations of the methods applied.

> This is a research prototype. It is not a medical device and must not be used
> for diagnosis or clinical decision-making.

CI runs linting, typing, tests, a complete quick-pipeline smoke test, package
build and container build on every change. Version tags create a GitHub
release with Python artifacts and publish the same versioned container to
`ghcr.io/ben120-web/ecg-motion-denoising`. Container publication is gated on
package validation, and ordinary branch pushes cannot publish release artifacts.
The registry release currently targets `linux/amd64`; ARM users can build the
same Dockerfile natively from source.

## Version 1.1.0

This release promotes the taught MSc modules from background material to
portfolio evidence. It adds constrained and bounded MPC implementations,
reproducible quadcopter and aircraft-pitch simulations, technical reports and
expanded regression coverage while preserving the complete denoising system
released in `v1.0.0`. The version identifies a software portfolio—not a
clinically approved model or flight-qualified controller.

## What this demonstrates

- Subject-grouped data splits to reduce identity and synthetic-source leakage.
- A hashed data manifest with schema, lineage and split cardinalities.
- A reproducible `prepare -> train -> evaluate` pipeline orchestrated by DVC.
- MLflow experiment tracking and TensorBoard training diagnostics.
- Configuration-driven PyTorch training with deterministic seeds.
- Scientific metrics: RMSE, normalised correlation and SNR improvement.
- Explicit aggregate and worst-case promotion checks with a recorded decision.
- A contract-validated FastAPI inference service with Prometheus metrics.
- Bounded quadcopter altitude MPC with solver diagnostics and closed-loop metrics.
- Constrained aircraft-pitch MPC enforcing the complete predicted flight envelope.
- Kalman filtering, DSP, Bayesian estimation, forecasting and sensor analysis.
- Unit tests, linting, type checks, CI and a non-root container runtime.

Explore the [MSc module showcase](coursework/README.md), or go directly to the
[quadcopter report](coursework/quadcopter_control/REPORT.md) and
[aircraft-pitch report](coursework/aircraft_mpc/REPORT.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Reproduce the complete local pipeline.
dvc repro

# Inspect results and experiments.
dvc metrics show
mlflow ui --backend-store-uri sqlite:///mlflow.db
tensorboard --logdir logs/tensorboard
```

Serve the trained checkpoint locally:

```bash
ecg-denoise serve --model-path models/model.pt
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/metrics
```

The versioned `POST /v1/denoise` endpoint accepts exactly one signal window and
its sampling rate. OpenAPI documentation is available at `/docs`. See the
[operations guide](docs/operations.md) for container use and the service
contract.

The default configuration generates a small deterministic synthetic dataset so
that the complete workflow can be exercised without access to private or large
research data. To use a DVC remote, see [the data guide](docs/data.md).

## Iterating safely

Change a value in [`params.yaml`](params.yaml), then run:

```bash
dvc repro
dvc metrics diff
```

Every training run records parameters, metrics and artifacts in local MLflow.
TensorBoard records batch/epoch curves. DVC decides which pipeline stages need
to be rerun and versions the resulting data/model artifacts independently of
Git.

## Reference result

The reproducible default CPU run improves mean held-out synthetic SNR by
**0.970 dB**, but reduces SNR by **4.483 dB** in the cleanest 24 dB stratum.
The machine-readable promotion decision is therefore **rejected**. This is the
intended governance behaviour: an aggregate gain does not conceal a harmful
slice. See the [reference report](reports/reference/README.md), its
[data manifest](reports/reference/data_manifest.json) and the
[promotion record](reports/reference/promotion.json).

These numbers validate the pipeline and expose the next modelling problem;
they are not evidence of clinical efficacy.

## Repository layout

```text
configs/                 named experiment overrides
coursework/              MSc module showcase, reports and control demonstrations
data/                    DVC-managed datasets
docs/                    data, experiment and deployment guidance
legacy/                  original MSc research implementation and MATLAB work
models/                  DVC-managed trained model artifacts
reports/                 machine-readable evaluation outputs
src/ecg_denoising/       maintained Python package
tests/                   fast unit and pipeline-contract tests
dvc.yaml                 reproducible pipeline graph
params.yaml              versioned experiment parameters
```

## Commands

```bash
make install     # install development dependencies
make pipeline    # run dvc repro
make test        # unit tests
make quality     # lint and type checks
make serve-api
make serve-mlflow
make tensorboard
```

## Research provenance

The original dissertation explored RCNN, convolutional denoising autoencoder
and recurrent architectures against wavelet and empirical-mode-decomposition
baselines. Original dissertation source retained for provenance lives under
`legacy/`; it is intentionally excluded from the maintained Python package.
Taught-module work is presented under [`coursework/`](coursework/README.md),
with maintained, tested implementations for the two MPC studies.

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use and limitations and
[`docs/experiments.md`](docs/experiments.md) for the experiment protocol. System
design and operational decisions are documented in
[`docs/architecture.md`](docs/architecture.md),
[`docs/algorithm.md`](docs/algorithm.md) and
[`docs/operations.md`](docs/operations.md).

## Author

Ben Russell — MSc Electronics Engineering

## License

Code is released under the MIT License. Dataset and third-party source licences
must be checked independently before redistribution.
