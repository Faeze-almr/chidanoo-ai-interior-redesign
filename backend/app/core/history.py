"""File-backed local generation history.

Each successful generation is an immutable folder with relative asset paths and
the canonical structured generation metadata.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.config import OUTPUTS_DIR, STORAGE_DIR
from app.core.errors import StorageError
from app.core.pipeline import GenerationResult

logger = logging.getLogger(__name__)

HISTORY_DIR = STORAGE_DIR / "history"
HISTORY_EXPORT_DIR = OUTPUTS_DIR / "history_exports"
_ENTRY_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}_[0-9a-f]{8}$")


def _history_root(root: str | Path | None = None) -> Path:
    path = Path(root) if root is not None else HISTORY_DIR
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError("The history folder could not be created.") from exc
    return path.resolve()


def _new_entry_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _save_optional_image(image: Image.Image | None, path: Path, mode: str) -> str | None:
    if image is None:
        return None
    image.convert(mode).save(path, format="PNG")
    return path.name


def save_history_entry(
    result: GenerationResult,
    input_image: Image.Image,
    *,
    style: str | None = None,
    user_request: str | None = None,
    reference_image: Image.Image | None = None,
    mask_image: Image.Image | None = None,
    root: str | Path | None = None,
) -> dict:
    """Atomically saves one successful canonical generation and its evidence."""

    history_root = _history_root(root)
    entry_id = _new_entry_id()
    final_dir = history_root / entry_id
    staging_dir = history_root / f".{entry_id}.tmp"

    try:
        staging_dir.mkdir(parents=False, exist_ok=False)
        input_name = _save_optional_image(input_image, staging_dir / "input.png", "RGB")
        output_name = _save_optional_image(result.image, staging_dir / "output.png", "RGB")
        reference_name = _save_optional_image(
            reference_image, staging_dir / "reference.png", "RGB"
        )
        mask_name = _save_optional_image(mask_image, staging_dir / "mask.png", "L")

        metadata = result.metadata.to_dict()
        record = {
            "id": entry_id,
            "created_at": metadata["timestamp"],
            "mode": metadata["mode"],
            "style": style,
            "user_request": user_request,
            "input_path": input_name,
            "output_path": output_name,
            "reference_path": reference_name,
            "mask_path": mask_name,
            "generation": metadata,
        }
        (staging_dir / "meta.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging_dir.replace(final_dir)
        return record
    except (OSError, TypeError, ValueError) as exc:
        logger.exception("Failed to save history entry %s", entry_id)
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise StorageError("The generation could not be added to local history.") from exc


def _entry_dir(entry_id: str, root: str | Path | None = None) -> Path:
    if not isinstance(entry_id, str) or not _ENTRY_ID_PATTERN.fullmatch(entry_id):
        raise StorageError("The selected history entry is invalid.")
    history_root = _history_root(root)
    entry_dir = (history_root / entry_id).resolve()
    if entry_dir.parent != history_root:
        raise StorageError("The selected history entry is invalid.")
    return entry_dir


def load_history_entry(entry_id: str, root: str | Path | None = None) -> dict:
    meta_path = _entry_dir(entry_id, root) / "meta.json"
    try:
        record = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError("The selected history entry could not be loaded.") from exc
    if record.get("id") != entry_id or not record.get("output_path"):
        raise StorageError("The selected history entry is incomplete.")
    return record


def load_history_output(entry_id: str, root: str | Path | None = None) -> Image.Image:
    entry_dir = _entry_dir(entry_id, root)
    record = load_history_entry(entry_id, root)
    output_path = (entry_dir / record["output_path"]).resolve()
    if output_path.parent != entry_dir:
        raise StorageError("The selected history output path is invalid.")
    try:
        with Image.open(output_path) as image:
            image.load()
            return image.convert("RGB")
    except OSError as exc:
        raise StorageError("The selected history image could not be opened.") from exc


def load_history_entries(limit: int = 100, root: str | Path | None = None) -> list[dict]:
    history_root = _history_root(root)
    safe_limit = max(0, min(int(limit), 1000))
    if safe_limit == 0:
        return []
    entries: list[dict] = []
    for meta_path in sorted(history_root.glob("*/meta.json"), reverse=True):
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
            if _ENTRY_ID_PATTERN.fullmatch(str(record.get("id", ""))):
                entries.append(record)
        except (OSError, json.JSONDecodeError, TypeError):
            logger.warning("Skipping malformed history metadata: %s", meta_path, exc_info=True)
        if len(entries) >= safe_limit:
            break
    return entries


def history_views(
    limit: int = 100,
    root: str | Path | None = None,
) -> tuple[list[tuple[str, str]], list[dict], list[str]]:
    history_root = _history_root(root)
    gallery: list[tuple[str, str]] = []
    rows: list[dict] = []
    choices: list[str] = []
    for record in load_history_entries(limit, history_root):
        generation = record.get("generation", {})
        output_name = str(record.get("output_path", ""))
        if not output_name or Path(output_name).name != output_name:
            logger.warning("Skipping history entry with an unsafe output path: %s", record["id"])
            continue
        output_path = history_root / record["id"] / output_name
        if not output_path.is_file():
            logger.warning("Skipping history entry with a missing output: %s", record["id"])
            continue
        caption = f"{record.get('mode')} | {record.get('style') or 'custom'} | {record.get('created_at')}"
        gallery.append((str(output_path), caption))
        rows.append(
            {
                "ID": record["id"],
                "Time": record.get("created_at"),
                "Mode": record.get("mode"),
                "Style": record.get("style"),
                "Prompt": generation.get("prompt"),
                "Seconds": generation.get("runtime_sec"),
                "Seed": generation.get("seed"),
            }
        )
        choices.append(record["id"])
    return gallery, rows, choices


def export_history_zip(
    root: str | Path | None = None,
    export_dir: str | Path | None = None,
) -> Path:
    history_root = _history_root(root)
    destination = Path(export_dir) if export_dir is not None else HISTORY_EXPORT_DIR
    try:
        destination.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output = destination / f"chidanoo_history_{timestamp}_{uuid.uuid4().hex[:6]}.zip"
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(history_root.rglob("*")):
                relative = path.relative_to(history_root)
                is_staging = any(
                    part.startswith(".") and part.endswith(".tmp") for part in relative.parts
                )
                if path.is_file() and not is_staging:
                    archive.write(path, relative)
        return output
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        logger.exception("Failed to export local history")
        raise StorageError("Local history could not be exported.") from exc
