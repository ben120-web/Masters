# Operations guide

## Local service

Produce a compatible checkpoint with `dvc repro`, then start the API:

```bash
ecg-denoise serve --model-path models/model.pt --host 0.0.0.0 --port 8000
```

- `GET /health/live` confirms the process is responsive.
- `GET /health/ready` confirms a checkpoint passed validation and is loaded.
- `POST /v1/denoise` accepts `sampling_rate_hz` and one `samples` array.
- `GET /metrics` exposes Prometheus request counters and latency histograms.

Requests with a wrong sample rate, wrong window length, non-finite values or a
malformed shape are rejected. Raw input signals are not logged.

## Container

The image runs as UID/GID `10001` and expects a read-only checkpoint mount:

```bash
docker build -t ecg-motion-denoising:local .
docker run --rm -p 8000:8000 \
  -v "$PWD/models/model.pt:/models/model.pt:ro" \
  ecg-motion-denoising:local
```

The container health check uses liveness. An orchestrator should use readiness
to keep an instance out of service until its model is valid.

The tagged registry image currently targets `linux/amd64`:

```bash
docker pull ghcr.io/ben120-web/ecg-motion-denoising:1.0.0
```

On an ARM development machine, build the Dockerfile locally for a native image
or run the registry image through explicit amd64 emulation.

## Release and rollback

Semantic version tags build Python artifacts and a GHCR image. Treat the image
digest and checkpoint version as one release record. Roll back by deploying the
previous image digest and its matching checkpoint; never silently replace a
checkpoint beneath a running instance.

Alert on sustained readiness failure, error-rate increase or latency regression.
Biomedical distribution and outcome monitoring require an approved data
governance design and are intentionally outside this synthetic demonstration.
