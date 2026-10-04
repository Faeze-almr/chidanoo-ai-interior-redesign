import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ModelLoadError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "gpu_smoke_test.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("gpu_smoke_test_for_tests", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_default_still_selects_all_five_cases():
    smoke = _load_smoke_module()
    args = smoke.parse_args([])
    assert args.cases == smoke.SMOKE_CASES


def test_smoke_accepts_reference_quick_debug_subset():
    smoke = _load_smoke_module()
    args = smoke.parse_args(["--cases", "reference,full_after_reference"])
    assert args.cases == ("reference", "full_after_reference")


def test_reference_smoke_exercises_notebook_fallback_prompt():
    smoke = _load_smoke_module()
    source = inspect.getsource(smoke.main)
    assert "use the reference image for materials and color mood" not in source


def test_room04_validation_fixture_keeps_nb09_coverage():
    fixture = PROJECT_ROOT / "test_fixtures" / "room_04_furniture_mask.png"
    with Image.open(fixture) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
        assert image.size == (1536, 1024)
    assert set(np.unique(mask)).issubset({0, 255})
    coverage = float(np.count_nonzero(mask >= 128) / mask.size * 100.0)
    assert coverage == pytest.approx(16.1343, abs=0.01)


def test_smoke_skips_full_reload_after_abnormal_reference_cleanup(monkeypatch, tmp_path):
    smoke = _load_smoke_module()
    room = tmp_path / "room.png"
    Image.new("RGB", (512, 512), "gray").save(room)
    output_dir = tmp_path / "outputs"
    args = SimpleNamespace(
        room08=room,
        room16=room,
        room04=room,
        reference_image=room,
        room04_mask=room,
        output_dir=output_dir,
        cases=("reference", "full_after_reference"),
    )
    monkeypatch.setattr(smoke, "parse_args", lambda: args)
    monkeypatch.setattr(
        smoke,
        "generate_reference_result",
        MagicMock(side_effect=ModelLoadError("Reference failed safely.")),
    )
    full_generation = MagicMock()
    monkeypatch.setattr(smoke, "generate_room_result", full_generation)
    monkeypatch.setattr(
        smoke.reference_adapter_manager,
        "_cleanup_allows_full_reload",
        False,
    )
    monkeypatch.setattr(
        smoke.reference_adapter_manager,
        "_last_cleanup_allocated",
        2 * 1024**3,
    )
    monkeypatch.setattr(
        smoke.reference_adapter_manager,
        "_last_cleanup_reserved",
        3 * 1024**3,
    )

    assert smoke.main() == 1
    full_generation.assert_not_called()
    records = json.loads((output_dir / "smoke_results.json").read_text(encoding="utf-8"))
    assert records[-1]["case"] == "room_08_full_after_reference"
    assert "abnormal CUDA allocation" in records[-1]["error"]
