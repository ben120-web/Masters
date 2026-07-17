"""FastAPI inference service with explicit health and telemetry contracts."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram
from prometheus_client.exposition import generate_latest
from pydantic import BaseModel, Field

from . import __version__
from .inference import DenoisingPredictor, InputContractError

LOGGER = logging.getLogger(__name__)


class DenoiseRequest(BaseModel):
    sampling_rate_hz: int = Field(gt=0)
    samples: Annotated[list[float], Field(min_length=1, max_length=100_000)]


class DenoiseResponse(BaseModel):
    sampling_rate_hz: int
    sample_count: int
    model_version: str
    denoised: list[float]


@dataclass
class ServiceState:
    predictor: DenoisingPredictor | None = None
    load_error: str | None = None


def create_app(
    model_path: str | Path | None = None,
    device: str | None = None,
) -> FastAPI:
    """Create an isolated service instance, suitable for production and tests."""
    resolved_model_value = (
        model_path if model_path is not None else os.getenv("ECG_MODEL_PATH", "models/model.pt")
    )
    resolved_model_path = Path(resolved_model_value)
    resolved_device = device if device is not None else os.getenv("ECG_DEVICE", "auto")
    state = ServiceState()
    registry = CollectorRegistry(auto_describe=True)
    requests = Counter(
        "ecg_denoise_requests_total",
        "Denoising requests partitioned by response status.",
        ("status",),
        registry=registry,
    )
    latency = Histogram(
        "ecg_denoise_request_duration_seconds",
        "Denoising request latency in seconds.",
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            state.predictor = DenoisingPredictor.from_checkpoint(
                resolved_model_path, resolved_device
            )
            state.load_error = None
            LOGGER.info("Loaded model from %s", resolved_model_path)
        except Exception as error:  # Service must remain live while readiness reports failure.
            state.predictor = None
            state.load_error = type(error).__name__
            LOGGER.exception("Model failed to load from %s", resolved_model_path)
        yield

    application = FastAPI(
        title="ECG motion-artefact denoising",
        version=__version__,
        lifespan=lifespan,
    )

    @application.get("/health/live")
    async def liveness() -> dict[str, str]:
        return {"status": "alive", "service_version": __version__}

    @application.get("/health/ready")
    async def readiness() -> dict[str, str]:
        if state.predictor is None:
            raise HTTPException(
                status_code=503,
                detail=f"model unavailable ({state.load_error or 'not loaded'})",
            )
        return {"status": "ready", "model_version": state.predictor.model_version}

    @application.post("/v1/denoise", response_model=DenoiseResponse)
    async def denoise(payload: DenoiseRequest) -> DenoiseResponse:
        started = perf_counter()
        if state.predictor is None:
            requests.labels(status="503").inc()
            raise HTTPException(status_code=503, detail="model unavailable")
        try:
            prediction = state.predictor.denoise(payload.samples, payload.sampling_rate_hz)
        except InputContractError as error:
            requests.labels(status="422").inc()
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            latency.observe(perf_counter() - started)
        requests.labels(status="200").inc()
        return DenoiseResponse(
            sampling_rate_hz=state.predictor.contract.sampling_rate_hz,
            sample_count=state.predictor.contract.samples,
            model_version=state.predictor.model_version,
            denoised=prediction.tolist(),
        )

    @application.get("/metrics", response_class=Response)
    async def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
