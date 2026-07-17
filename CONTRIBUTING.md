# Contributing

Create a focused branch and keep maintained changes outside `legacy/` and
`coursework/`. Install the development environment with:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Before opening a pull request, run:

```bash
make quality
make test
python -m ecg_denoising.cli pipeline --config configs/quick.yaml
```

Changes to data preparation must retain subject and source isolation, update the
data contract tests and avoid committing raw biomedical data. Changes to model
inputs or checkpoint structure must update inference, API and model-card
documentation together. Include tests for failure paths as well as the expected
path, and explain any metric trade-off introduced by an algorithm change.

Use clear commit messages and never include secrets, absolute personal paths or
third-party data without an explicit redistribution licence.
