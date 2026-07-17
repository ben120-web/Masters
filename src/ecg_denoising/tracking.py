"""Thin tracking adapters so core training remains testable."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, uri: str, experiment_name: str) -> None:
        self.mlflow: Any | None = None
        try:
            import mlflow

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            self.mlflow = mlflow
        except ImportError:
            LOGGER.warning(
                "MLflow is unavailable; experiment metadata will only be written locally"
            )

    def run(self, run_name: str) -> AbstractContextManager[Any]:
        if self.mlflow is None:
            return nullcontext()
        return self.mlflow.start_run(run_name=run_name)

    def log_params(self, values: dict[str, Any]) -> None:
        if self.mlflow is not None:
            self.mlflow.log_params(values)

    def log_metrics(self, values: dict[str, float], step: int | None = None) -> None:
        if self.mlflow is not None:
            self.mlflow.log_metrics(values, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self.mlflow is not None:
            self.mlflow.log_artifact(str(path))


class TensorBoardTracker:
    def __init__(self, directory: str | Path) -> None:
        self.writer: Any | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(directory))
        except ImportError:
            LOGGER.warning("TensorBoard is unavailable; scalar summaries are disabled")

    def scalar(self, name: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
