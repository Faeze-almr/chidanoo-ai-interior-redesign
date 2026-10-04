"""End-to-end API tests for the full-room redesign workflow.

Exercises the real HTTP route -> job queue -> app.core.pipeline.redesign_room
-> app.core.model_manager chain exactly as it runs in production, except
``model_manager.get_pipe``/``get_midas`` are monkeypatched with lightweight
fakes (same approach as tests/test_pipeline.py) so no SDXL weights are
downloaded and no GPU inference runs. Upload/result storage is redirected
to a pytest tmp_path so these tests never touch backend/storage/.
"""

import io
import json
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes_generate
from app.core import pipeline, storage
from app.core.errors import StorageError
from app.core.model_manager import model_manager, require_cuda as enforce_cuda
from app.main import app


@pytest.fixture(autouse=True)
def allow_mocked_generation_without_cuda(monkeypatch):
    monkeypatch.setattr(routes_generate, "require_cuda", lambda: None)
    monkeypatch.setattr(pipeline, "require_cuda", lambda: None)


def _valid_image_bytes(size=(512, 512)) -> bytes:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[: size[1] // 2, :, :] = 40
    arr[size[1] // 2 :, :, :] = 200
    image = Image.fromarray(arr)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _too_small_image_bytes() -> bytes:
    image = Image.new("RGB", (100, 100), (128, 128, 128))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class FakeMidas:
    def __call__(self, image):
        return image


class FakePipe:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        from types import SimpleNamespace

        return SimpleNamespace(images=[Image.new("RGB", (64, 64), (5, 10, 15))])


def _redirect_storage_to_tmp(monkeypatch, tmp_path):
    uploads_dir = tmp_path / "uploads"
    results_dir = tmp_path / "results"
    uploads_dir.mkdir()
    results_dir.mkdir()
    monkeypatch.setattr(storage, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(storage, "RESULTS_DIR", results_dir)


def _poll_until_finished(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout_s}s")


def test_generate_room_rejects_too_small_image_with_400(monkeypatch, tmp_path):
    _redirect_storage_to_tmp(monkeypatch, tmp_path)

    with TestClient(app) as client:
        resp = client.post(
            "/api/generate/room",
            files={"image": ("room.png", _too_small_image_bytes(), "image/png")},
        )

    assert resp.status_code == 400
    assert "small" in resp.json()["detail"].lower()


def test_generate_room_rejects_non_image_upload_with_400(monkeypatch, tmp_path):
    _redirect_storage_to_tmp(monkeypatch, tmp_path)

    with TestClient(app) as client:
        resp = client.post(
            "/api/generate/room",
            files={"image": ("not-an-image.txt", b"this is not an image", "text/plain")},
        )

    assert resp.status_code == 400
    assert "not a readable image" in resp.json()["detail"].lower()


def test_job_status_404_for_unknown_job_id(monkeypatch, tmp_path):
    _redirect_storage_to_tmp(monkeypatch, tmp_path)

    with TestClient(app) as client:
        resp = client.get("/api/jobs/does-not-exist")

    assert resp.status_code == 404


def test_generate_room_happy_path_end_to_end(monkeypatch, tmp_path):
    _redirect_storage_to_tmp(monkeypatch, tmp_path)
    fake_pipe = FakePipe()
    monkeypatch.setattr(model_manager, "get_pipe", lambda: fake_pipe)
    monkeypatch.setattr(model_manager, "get_midas", lambda: FakeMidas())

    with TestClient(app) as client:
        submit_resp = client.post(
            "/api/generate/room",
            files={"image": ("room.png", _valid_image_bytes(), "image/png")},
            data={
                "style_key": "industrial",
                "user_text": "exposed brick and a leather sofa",
                "depth_scale": "0.6",
                "canny_scale": "0.15",
            },
        )
        assert submit_resp.status_code == 202
        submitted = submit_resp.json()
        assert submitted["status"] == "queued"
        job_id = submitted["job_id"]

        final = _poll_until_finished(client, job_id)
        assert final["status"] == "succeeded"
        assert final["result_url"] == f"/api/jobs/{job_id}/image"
        assert final["error"] is None
        assert final["started_at"] is not None
        assert final["finished_at"] is not None

        image_resp = client.get(final["result_url"])
        assert image_resp.status_code == 200
        assert image_resp.headers["content-type"] == "image/png"

        download_resp = client.get(final["result_url"], params={"download": "true"})
        assert download_resp.status_code == 200
        assert "attachment" in download_resp.headers["content-disposition"]
        assert f"chidanoo-{job_id}.png" in download_resp.headers["content-disposition"]

    # The real (unmodified) redesign_room ran and called our fake pipe with
    # the request's actual parameters - proves the whole chain is wired up,
    # not just the HTTP shell around it.
    assert len(fake_pipe.calls) == 1
    call = fake_pipe.calls[0]
    assert call["controlnet_conditioning_scale"] == [0.6, 0.15]
    assert "exposed brick and a leather sofa" in call["prompt"]
    metadata_path = storage.RESULTS_DIR / f"{job_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["mode"] == "full_redesign"
    assert metadata["seed"] == 42


def test_generate_room_failure_is_reported_not_swallowed(monkeypatch, tmp_path):
    _redirect_storage_to_tmp(monkeypatch, tmp_path)

    def broken_pipe(**kwargs):
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr(model_manager, "get_pipe", lambda: broken_pipe)
    monkeypatch.setattr(model_manager, "get_midas", lambda: FakeMidas())

    with TestClient(app) as client:
        submit_resp = client.post(
            "/api/generate/room",
            files={"image": ("room.png", _valid_image_bytes(), "image/png")},
        )
        job_id = submit_resp.json()["job_id"]

        final = _poll_until_finished(client, job_id)
        assert final["status"] == "failed"
        assert final["error"] == "Generation failed while processing the image."
        assert final["result_url"] is None

        missing_image_resp = client.get(f"/api/jobs/{job_id}/image")
        assert missing_image_resp.status_code == 404


def test_generate_room_rejects_invalid_numeric_parameter(monkeypatch, tmp_path):
    _redirect_storage_to_tmp(monkeypatch, tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/generate/room",
            files={"image": ("room.png", _valid_image_bytes(), "image/png")},
            data={"depth_scale": "1.5"},
        )
    assert response.status_code == 400
    assert "between 0 and 1" in response.json()["detail"]


def test_generate_room_requires_cuda_before_creating_job(monkeypatch, tmp_path):
    _redirect_storage_to_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(routes_generate, "require_cuda", enforce_cuda)
    with TestClient(app) as client:
        response = client.post(
            "/api/generate/room",
            files={"image": ("room.png", _valid_image_bytes(), "image/png")},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "GPU/CUDA is required for this generation pipeline."


def test_upload_save_failure_returns_safe_error(monkeypatch, tmp_path):
    _redirect_storage_to_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(
        routes_generate,
        "save_upload_image",
        lambda *args, **kwargs: (_ for _ in ()).throw(StorageError("The uploaded image could not be saved.")),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/generate/room",
            files={"image": ("room.png", _valid_image_bytes(), "image/png")},
        )
    assert response.status_code == 500
    assert response.json()["detail"] == "The uploaded image could not be saved."
