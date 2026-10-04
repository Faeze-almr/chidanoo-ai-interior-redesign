from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes_history
from app.core import history
from app.core.pipeline import GenerationMetadata, GenerationResult
from app.main import app


def _result() -> GenerationResult:
    metadata = GenerationMetadata(
        mode="full_redesign",
        timestamp="2026-09-22T10:00:00+00:00",
        model="test/model",
        seed=42,
        steps=30,
        cfg=7.5,
        depth_scale=0.5,
        canny_scale=0.2,
        resolution=(16, 16),
        runtime_sec=1.0,
        peak_vram_gb=1.0,
        prompt="modern room",
        negative_prompt="artifacts",
        status="succeeded",
    )
    return GenerationResult(Image.new("RGB", (16, 16), "blue"), metadata)


def test_history_list_asset_and_export(monkeypatch, tmp_path):
    history_root = tmp_path / "history"
    export_root = tmp_path / "exports"
    monkeypatch.setattr(history, "HISTORY_DIR", history_root)
    monkeypatch.setattr(history, "HISTORY_EXPORT_DIR", export_root)
    monkeypatch.setattr(routes_history, "HISTORY_DIR", history_root)

    record = history.save_history_entry(
        _result(),
        Image.new("RGB", (16, 16), "gray"),
        style="modern",
        user_request="warm wood",
    )

    with TestClient(app) as client:
        listed = client.get("/api/history")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == record["id"]

        image = client.get(f"/api/history/{record['id']}/output")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"

        archive = client.get("/api/history/export")
        assert archive.status_code == 200
        assert archive.headers["content-type"] == "application/zip"


def test_history_asset_rejects_unknown_asset(monkeypatch, tmp_path):
    history_root = tmp_path / "history"
    monkeypatch.setattr(history, "HISTORY_DIR", history_root)
    monkeypatch.setattr(routes_history, "HISTORY_DIR", history_root)
    with TestClient(app) as client:
        response = client.get("/api/history/invalid/unknown")
    assert response.status_code == 404

