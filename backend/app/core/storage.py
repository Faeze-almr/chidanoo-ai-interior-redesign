"""Local filesystem storage for transient job uploads and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import STORAGE_DIR
from app.core.errors import StorageError

UPLOADS_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_upload_image(image: Image.Image, job_id: str) -> Path:
    path = UPLOADS_DIR / f"{job_id}.png"
    try:
        image.convert("RGB").save(path, format="PNG")
    except OSError as exc:
        raise StorageError("The uploaded image could not be saved.") from exc
    return path


def save_result_image(image: Image.Image, job_id: str) -> Path:
    path = RESULTS_DIR / f"{job_id}.png"
    try:
        image.convert("RGB").save(path, format="PNG")
    except OSError as exc:
        raise StorageError("The generated image could not be saved.") from exc
    return path


def save_generation_metadata(metadata: Any, job_id: str) -> Path:
    """Writes a JSON sidecar without introducing a history/database layer."""

    path = RESULTS_DIR / f"{job_id}.json"
    payload = metadata.to_dict() if hasattr(metadata, "to_dict") else metadata
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise StorageError("Generation metadata could not be saved.") from exc
    return path
