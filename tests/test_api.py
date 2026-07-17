from fastapi.testclient import TestClient

from ecg_denoising import __version__
from ecg_denoising.api import create_app
from ecg_denoising.config import load_config


def test_service_health_inference_and_metrics(model_checkpoint) -> None:
    config = load_config("configs/quick.yaml")
    with TestClient(create_app(model_checkpoint, "cpu")) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        response = client.post(
            "/v1/denoise",
            json={
                "sampling_rate_hz": config.data.sampling_rate_hz,
                "samples": [0.0] * config.data.samples,
            },
        )
        metrics = client.get("/metrics")

    assert live.json() == {"status": "alive", "service_version": __version__}
    assert ready.status_code == 200
    assert ready.json()["model_version"] == __version__
    assert response.status_code == 200
    assert response.json()["sample_count"] == config.data.samples
    assert len(response.json()["denoised"]) == config.data.samples
    assert 'ecg_denoise_requests_total{status="200"} 1.0' in metrics.text


def test_service_reports_unready_without_model(tmp_path) -> None:
    with TestClient(create_app(tmp_path / "missing.pt", "cpu")) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert (
            client.post("/v1/denoise", json={"sampling_rate_hz": 250, "samples": [0.0]}).status_code
            == 503
        )


def test_service_rejects_wrong_sampling_rate(model_checkpoint) -> None:
    config = load_config("configs/quick.yaml")
    with TestClient(create_app(model_checkpoint, "cpu")) as client:
        response = client.post(
            "/v1/denoise",
            json={
                "sampling_rate_hz": config.data.sampling_rate_hz + 1,
                "samples": [0.0] * config.data.samples,
            },
        )

    assert response.status_code == 422
    assert "sampling_rate_hz" in response.json()["detail"]
