from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from app.core import pipeline
from app.core.errors import (
    FeatureCleanupError,
    FeatureLoadError,
    GenerationOutOfMemoryError,
    InputValidationError,
    ModelLoadError,
    ParameterValidationError,
)
from app.core.final_config import REFERENCE_FEATURE_CONFIG, REFERENCE_REDESIGN_CONFIG
from app.core.ip_adapter import reference_adapter_manager
from app.core.model_manager import ModelManager, model_manager


@pytest.fixture(autouse=True)
def isolated_mocked_generation(monkeypatch):
    ModelManager.unload_all(model_manager)
    monkeypatch.setattr(pipeline, "require_cuda", lambda: None)
    yield
    ModelManager.unload_all(model_manager)


def _valid_image(size=(512, 512), invert=False):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[: size[1] // 2] = 200 if invert else 40
    arr[size[1] // 2 :] = 40 if invert else 200
    return Image.fromarray(arr)


class FakeMidas:
    def __call__(self, image):
        return image


class DefaultAttnProcessor:
    pass


class FakeUnet:
    def __init__(self):
        self.attn_processors = {"original": DefaultAttnProcessor()}
        self.encoder_hid_proj = None
        self.config = SimpleNamespace(encoder_hid_dim_type=None)


class FakeReferencePipe:
    def __init__(
        self,
        error=None,
        fail_load=False,
        fail_unload=False,
        encoder_projection_dim=1024,
        adapter_input_dim=1024,
    ):
        self.unet = FakeUnet()
        self.image_encoder = None
        self.feature_extractor = None
        self.calls = []
        self.events = []
        self.scale = None
        self.error = error
        self.fail_load = fail_load
        self.fail_unload = fail_unload
        self.unload_calls = 0
        self.encoder_projection_dim = encoder_projection_dim
        self.adapter_input_dim = adapter_input_dim
        self.hooks_freed = False
        self.hooks_removed = False
        self.components = {}

    def disable_attention_slicing(self):
        self.events.append("disable_slicing")

    def load_ip_adapter(self, *args, **kwargs):
        self.events.append("load_adapter")
        self.unet.attn_processors = {"adapter": object()}
        self.unet.encoder_hid_proj = SimpleNamespace(
            image_projection_layers=[
                SimpleNamespace(
                    image_embeds=SimpleNamespace(in_features=self.adapter_input_dim)
                )
            ]
        )
        self.image_encoder = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=1280,
                projection_dim=self.encoder_projection_dim,
            )
        )
        self.feature_extractor = object()
        if self.fail_load:
            raise RuntimeError("partial adapter failure")

    def set_ip_adapter_scale(self, scale):
        self.events.append("set_scale")
        self.scale = scale

    def enable_model_cpu_offload(self):
        self.events.append("cpu_offload")

    def unload_ip_adapter(self):
        self.events.append("unload_adapter")
        self.unload_calls += 1
        if self.fail_unload:
            raise RuntimeError("cleanup detail")

    def maybe_free_model_hooks(self):
        self.hooks_freed = True

    def remove_all_hooks(self):
        self.hooks_removed = True

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        size = kwargs["image"][0].size
        return SimpleNamespace(images=[Image.new("RGB", size, (30, 90, 150))])


class FakeCanonicalPipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        size = kwargs["image"][0].size
        return SimpleNamespace(images=[Image.new("RGB", size, (80, 100, 120))])


def _install_ephemeral(monkeypatch, fake_pipe, events):
    original_unload = ModelManager.unload_all

    def unload_all():
        events.append("unload_all")
        original_unload(model_manager)

    def create_reference_pipe():
        events.append("create_reference_pipe")
        assert model_manager.is_pipe_loaded is False
        assert model_manager.is_inpaint_pipe_loaded is False
        return fake_pipe

    def get_midas():
        events.append("get_midas")
        midas = FakeMidas()
        model_manager._midas = midas
        return midas

    monkeypatch.setattr(model_manager, "unload_all", unload_all)
    monkeypatch.setattr(model_manager, "create_reference_pipe", create_reference_pipe)
    monkeypatch.setattr(model_manager, "get_midas", get_midas)


def test_reference_is_ephemeral_cleans_manager_and_full_reloads_canonical(monkeypatch):
    fake_reference = FakeReferencePipe()
    events = []
    model_manager._pipe = object()
    model_manager._inpaint_pipe = object()
    model_manager._midas = object()
    _install_ephemeral(monkeypatch, fake_reference, events)

    result = pipeline.generate_reference_result(
        _valid_image(),
        _valid_image(invert=True),
        user_text="warm oak palette",
    )
    reference_call = fake_reference.calls[0]
    assert events[:3] == ["unload_all", "create_reference_pipe", "get_midas"]
    assert events[-1] == "unload_all"
    assert events.count("unload_all") == 2
    assert reference_call["controlnet_conditioning_scale"] == [0.50, 0.20]
    assert reference_call["num_inference_steps"] == 30
    assert reference_call["guidance_scale"] == 7.5
    assert reference_call["generator"].initial_seed() == 42
    assert reference_call["ip_adapter_image"].size == (512, 512)
    assert "a fully renovated real interior room" in reference_call["prompt"]
    assert "completely new furniture and decor" in reference_call["prompt"]
    assert "warm oak palette" in reference_call["prompt"]
    assert fake_reference.scale == 0.60
    assert fake_reference.events[:4] == [
        "disable_slicing",
        "load_adapter",
        "set_scale",
        "cpu_offload",
    ]
    assert fake_reference.unload_calls == 1
    assert fake_reference.image_encoder is None
    assert fake_reference.feature_extractor is None
    assert fake_reference.hooks_freed is True
    assert fake_reference.hooks_removed is True
    assert result.metadata.mode == "reference_redesign"
    assert result.metadata.reference_scale == REFERENCE_FEATURE_CONFIG.default_scale
    assert result.metadata.seed == REFERENCE_REDESIGN_CONFIG.seed
    assert reference_adapter_manager.is_loaded is False
    assert model_manager.is_pipe_loaded is False
    assert model_manager.is_inpaint_pipe_loaded is False
    assert model_manager.is_midas_loaded is False

    canonical = FakeCanonicalPipe()
    get_pipe = MagicMock(return_value=canonical)
    monkeypatch.setattr(model_manager, "get_pipe", get_pipe)
    monkeypatch.setattr(model_manager, "get_midas", lambda: FakeMidas())
    full = pipeline.generate_room_result(_valid_image(), style_key="modern")
    assert full.metadata.mode == "full_redesign"
    get_pipe.assert_called_once_with()
    assert len(canonical.calls) == 1
    assert len(fake_reference.calls) == 1


def test_reference_missing_image_and_invalid_scale_fail_before_models(monkeypatch):
    create_pipe = MagicMock(side_effect=AssertionError("model should not load"))
    monkeypatch.setattr(model_manager, "create_reference_pipe", create_pipe)
    with pytest.raises(InputValidationError):
        pipeline.generate_reference_result(_valid_image(), None)
    with pytest.raises(ParameterValidationError, match="Reference scale"):
        pipeline.generate_reference_result(_valid_image(), _valid_image(), reference_scale=1.1)
    create_pipe.assert_not_called()


def test_reference_oom_cleans_ephemeral_pipe_and_model_manager(monkeypatch):
    fake_pipe = FakeReferencePipe(error=torch.cuda.OutOfMemoryError("allocator detail"))
    events = []
    _install_ephemeral(monkeypatch, fake_pipe, events)
    with pytest.raises(GenerationOutOfMemoryError, match="GPU ran out of memory"):
        pipeline.generate_reference_result(_valid_image(), _valid_image(invert=True))
    assert fake_pipe.unload_calls == 1
    assert fake_pipe.image_encoder is None
    assert fake_pipe.hooks_freed is True
    assert fake_pipe.hooks_removed is True
    assert reference_adapter_manager.is_loaded is False
    assert model_manager.is_pipe_loaded is False
    assert model_manager.is_midas_loaded is False


def test_partial_adapter_load_failure_does_not_unload_and_manager_is_clean(monkeypatch):
    fake_pipe = FakeReferencePipe(fail_load=True)
    events = []
    _install_ephemeral(monkeypatch, fake_pipe, events)
    with pytest.raises(ModelLoadError, match="reference-image adapter"):
        pipeline.generate_reference_result(_valid_image(), _valid_image(invert=True))
    assert fake_pipe.unload_calls == 0
    assert fake_pipe.image_encoder is None
    assert fake_pipe.feature_extractor is None
    assert fake_pipe.hooks_freed is True
    assert fake_pipe.hooks_removed is True
    assert reference_adapter_manager.is_loaded is False
    assert model_manager.is_pipe_loaded is False
    assert model_manager.is_inpaint_pipe_loaded is False
    assert model_manager.is_midas_loaded is False


def test_incompatible_encoder_fails_before_diffusion_and_cleans_manager(monkeypatch):
    fake_pipe = FakeReferencePipe(encoder_projection_dim=1280, adapter_input_dim=1024)
    events = []
    _install_ephemeral(monkeypatch, fake_pipe, events)
    with pytest.raises(FeatureLoadError, match="projection_dim=1280, expected 1024"):
        pipeline.generate_reference_result(_valid_image(), _valid_image(invert=True))
    assert fake_pipe.calls == []
    assert fake_pipe.unload_calls == 1
    assert fake_pipe.hooks_freed is True
    assert fake_pipe.hooks_removed is True
    assert reference_adapter_manager.is_loaded is False
    assert model_manager.is_pipe_loaded is False
    assert model_manager.is_inpaint_pipe_loaded is False
    assert model_manager.is_midas_loaded is False


def test_reference_cleanup_failure_still_cleans_model_manager(monkeypatch):
    fake_pipe = FakeReferencePipe(fail_unload=True)
    events = []
    _install_ephemeral(monkeypatch, fake_pipe, events)
    with pytest.raises(FeatureCleanupError):
        pipeline.generate_reference_result(_valid_image(), _valid_image(invert=True))
    assert fake_pipe.unload_calls == 1
    assert fake_pipe.image_encoder is None
    assert reference_adapter_manager.is_loaded is False
    assert model_manager.is_pipe_loaded is False
    assert model_manager.is_midas_loaded is False
