import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from app.core.errors import StorageError
from app.core.history import (
    export_history_zip,
    history_views,
    load_history_entries,
    load_history_output,
    save_history_entry,
)
from app.core.pipeline import GenerationMetadata, GenerationResult


def _result():
    metadata = GenerationMetadata(
        mode="targeted_inpaint",
        timestamp="2026-09-22T10:00:00+00:00",
        model="test/model",
        seed=42,
        steps=30,
        cfg=7.5,
        depth_scale=0.4,
        canny_scale=0.1,
        strength=1.0,
        resolution=(32, 32),
        runtime_sec=1.25,
        peak_vram_gb=2.5,
        prompt="protected prompt",
        negative_prompt="negative",
        status="succeeded",
        mask_coverage=0.5,
        outside_mask_mae=0.0,
    )
    return GenerationResult(Image.new("RGB", (32, 32), "blue"), metadata)


def test_history_persists_canonical_metadata_and_optional_evidence(tmp_path):
    root = tmp_path / "history"
    source = Image.new("RGB", (32, 32), "gray")
    mask = Image.new("L", (32, 32), 255)
    reference = Image.new("RGB", (32, 32), "green")
    record = save_history_entry(
        _result(),
        source,
        style="modern",
        user_request="new bed",
        reference_image=reference,
        mask_image=mask,
        root=root,
    )

    entry_dir = root / record["id"]
    assert (entry_dir / "input.png").exists()
    assert (entry_dir / "output.png").exists()
    assert (entry_dir / "reference.png").exists()
    assert (entry_dir / "mask.png").exists()
    persisted = json.loads((entry_dir / "meta.json").read_text(encoding="utf-8"))
    assert persisted["generation"]["outside_mask_mae"] == 0.0
    assert persisted["generation"]["seed"] == 42
    assert load_history_output(record["id"], root).getpixel((0, 0)) == (0, 0, 255)


def test_history_views_and_zip_export(tmp_path):
    root = tmp_path / "history"
    record = save_history_entry(
        _result(), Image.new("RGB", (32, 32), "gray"), root=root
    )
    gallery, rows, choices = history_views(root=root)
    assert choices == [record["id"]]
    assert rows[0]["Seed"] == 42
    assert Path(gallery[0][0]).exists()

    archive = export_history_zip(root=root, export_dir=tmp_path / "exports")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
    assert any(name.endswith("meta.json") for name in names)
    assert any(name.endswith("output.png") for name in names)


def test_history_rejects_traversal_and_skips_malformed_metadata(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    malformed = root / "20260922T100000_deadbeef"
    malformed.mkdir()
    (malformed / "meta.json").write_text("not json", encoding="utf-8")
    assert load_history_entries(root=root) == []
    with pytest.raises(StorageError, match="invalid"):
        load_history_output("../outside", root)
