FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ECG_MODEL_PATH=/models/model.pt \
    ECG_DEVICE=cpu

LABEL org.opencontainers.image.title="ECG motion-artefact denoising" \
      org.opencontainers.image.description="Validated ECG denoising inference service" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install ".[serve]"

COPY params.yaml ./params.yaml
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /models /app/data /app/reports /app/logs \
    && chown -R app:app /models /app

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]

ENTRYPOINT ["ecg-denoise"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--model-path", "/models/model.pt", "--device", "cpu"]
