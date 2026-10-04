"""Import-safety and lazy-loading-structure tests.

These deliberately never call get_pipe()/get_inpaint_pipe()/get_midas() for
real - doing so would download multi-gigabyte SDXL weights. They only check
that importing the module and constructing a fresh manager never touches
the network or the GPU, and that the lazy-loading bookkeeping behaves as
expected when loader internals are stubbed out.
"""

import importlib

import pytest
import torch

from app.core.errors import CudaRequiredError, ModelLoadError
from app.core.model_manager import ModelManager, get_device, get_dtype, model_manager, require_cuda


def test_import_does_not_eagerly_load_any_model():
    # Re-import to make extra sure module-level code (not just __init__)
    # hasn't loaded anything as a side effect of being imported.
    module = importlib.import_module("app.core.model_manager")
    importlib.reload(module)
    assert module.model_manager.is_pipe_loaded is False
    assert module.model_manager.is_inpaint_pipe_loaded is False
    assert module.model_manager.is_midas_loaded is False


def test_fresh_manager_has_nothing_loaded():
    manager = ModelManager()
    assert manager.is_pipe_loaded is False
    assert manager.is_inpaint_pipe_loaded is False
    assert manager.is_midas_loaded is False


def test_module_level_singleton_starts_unloaded():
    assert model_manager.is_pipe_loaded is False
    assert model_manager.is_inpaint_pipe_loaded is False
    assert model_manager.is_midas_loaded is False


def test_get_device_matches_torch_cuda_availability():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device() == expected


def test_get_dtype_is_fp16_on_cuda_fp32_otherwise():
    assert get_dtype("cuda") == torch.float16
    assert get_dtype("cpu") == torch.float32


def test_require_cuda_fails_with_domain_message_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CudaRequiredError, match="GPU/CUDA is required"):
        require_cuda()


def test_get_pipe_is_lazy_and_cached(monkeypatch):
    """Stubs out the real loader so no network/GPU work happens, and
    verifies the manager only calls it once (caching) and only when asked."""
    manager = ModelManager()
    sentinel = object()
    calls = []

    def fake_load_pipe():
        calls.append(1)
        return sentinel

    monkeypatch.setattr(manager, "_load_pipe", fake_load_pipe)

    assert manager.is_pipe_loaded is False
    assert len(calls) == 0

    result_1 = manager.get_pipe()
    assert result_1 is sentinel
    assert manager.is_pipe_loaded is True
    assert len(calls) == 1

    result_2 = manager.get_pipe()
    assert result_2 is sentinel
    assert len(calls) == 1  # not reloaded on second call


def test_model_load_failure_is_wrapped_safely(monkeypatch):
    manager = ModelManager()
    monkeypatch.setattr(manager, "_load_pipe", lambda: (_ for _ in ()).throw(OSError("secret path")))
    with pytest.raises(ModelLoadError, match="could not be loaded"):
        manager.get_pipe()


def test_get_inpaint_pipe_builds_from_the_main_pipe(monkeypatch):
    """Verifies the final shared-component design: the inpaint pipe is built
    from the main pipe's own components, not loaded independently."""
    manager = ModelManager()
    fake_pipe = object()
    build_calls = []

    monkeypatch.setattr(manager, "_load_pipe", lambda: fake_pipe)

    def fake_build_inpaint_pipe(pipe):
        build_calls.append(pipe)
        return "fake-inpaint-pipe"

    monkeypatch.setattr(manager, "_build_inpaint_pipe", fake_build_inpaint_pipe)

    result = manager.get_inpaint_pipe()
    assert result == "fake-inpaint-pipe"
    assert build_calls == [fake_pipe]
    assert manager.is_pipe_loaded is True  # loading the inpaint pipe also loads the main pipe


def test_reference_pipe_factory_never_populates_canonical_cache(monkeypatch):
    manager = ModelManager()
    created = [object(), object()]
    monkeypatch.setattr(manager, "_build_reference_pipe", lambda: created.pop(0))

    first = manager.create_reference_pipe()
    second = manager.create_reference_pipe()

    assert first is not second
    assert manager.is_pipe_loaded is False
    assert manager.is_inpaint_pipe_loaded is False
    assert manager.is_midas_loaded is False


def test_unload_all_clears_every_component(monkeypatch):
    manager = ModelManager()
    monkeypatch.setattr(manager, "_load_pipe", lambda: object())
    monkeypatch.setattr(manager, "_load_midas", lambda: object())

    manager.get_pipe()
    manager.get_midas()
    assert manager.is_pipe_loaded is True
    assert manager.is_midas_loaded is True

    manager.unload_all()

    assert manager.is_pipe_loaded is False
    assert manager.is_inpaint_pipe_loaded is False
    assert manager.is_midas_loaded is False
