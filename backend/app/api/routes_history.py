"""Read-only API for the existing file-backed generation history."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.errors import StorageError
from app.core.history import (
    HISTORY_DIR,
    export_history_zip,
    load_history_entries,
    load_history_entry,
)

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
def list_history(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
    try:
        return load_history_entries(limit=limit)
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=exc.user_message) from exc


@router.get("/export")
def download_history_export() -> FileResponse:
    try:
        archive = export_history_zip()
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=exc.user_message) from exc
    return FileResponse(
        archive,
        media_type="application/zip",
        filename=archive.name,
        content_disposition_type="attachment",
    )


@router.get("/{entry_id}/{asset}")
def get_history_asset(entry_id: str, asset: str, download: bool = False) -> FileResponse:
    field_by_asset = {
        "input": "input_path",
        "output": "output_path",
        "reference": "reference_path",
        "mask": "mask_path",
    }
    field = field_by_asset.get(asset)
    if field is None:
        raise HTTPException(status_code=404, detail="History asset not found.")
    try:
        record = load_history_entry(entry_id)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=exc.user_message) from exc
    filename = record.get(field)
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=404, detail="History asset not found.")
    entry_dir = (HISTORY_DIR / entry_id).resolve()
    path = (entry_dir / filename).resolve()
    if path.parent != entry_dir or not path.is_file():
        raise HTTPException(status_code=404, detail="History asset not found.")
    media_type = "image/png"
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"chidanoo-{entry_id}-{asset}.png" if download else None,
        content_disposition_type="attachment" if download else "inline",
    )
