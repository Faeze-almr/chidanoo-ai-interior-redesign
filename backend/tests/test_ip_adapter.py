from types import SimpleNamespace

import pytest

from app.core.errors import FeatureCleanupError, FeatureLoadError, ModelLoadError
from app.core.ip_adapter import AdapterLoadState, ReferenceAdapterManager


class DefaultAttnProcessor:
    pass


class SlicedAttnProcessor:
    pass


class FakeUnet:
    def __init__(self, *, sliced=False):
        processor = SlicedAttnProcessor() if sliced else DefaultAttnProcessor()
        self.attn_processors = {"down": processor}
        self.encoder_hid_proj = None
        self.config = SimpleNamespace(encoder_hid_dim_type=None)


class FakePipe:
    def __init__(
        self,
        *,
        fail_load=False,
        fail_unload=False,
        sliced=False,
        retain_sliced=False,
        encoder_projection_dim=1024,
        adapter_input_dim=1024,
    ):
        self.unet = FakeUnet(sliced=sliced)
        self.image_encoder = None
        self.feature_extractor = None
        self.fail_load = fail_load
        self.fail_unload = fail_unload
        self.retain_sliced = retain_sliced
        self.events = []
        self.load_kwargs = None
        self.scale = None
        self.unload_calls = 0
        self.encoder_projection_dim = encoder_projection_dim
        self.adapter_input_dim = adapter_input_dim
        self.hooks_freed = False
        self.hooks_removed = False
        self.components = {}

    def disable_attention_slicing(self):
        self.events.append("disable_slicing")
        if not self.retain_sliced:
            self.unet.attn_processors = {"down": DefaultAttnProcessor()}

    def load_ip_adapter(self, *args, **kwargs):
        self.events.append("load_adapter")
        assert not any(
            type(processor).__name__.startswith("SlicedAttnProcessor")
            for processor in self.unet.attn_processors.values()
        )
        self.load_kwargs = (args, kwargs)
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
            raise RuntimeError("private adapter load detail")

    def set_ip_adapter_scale(self, scale):
        self.events.append("set_scale")
        self.scale = scale

    def enable_model_cpu_offload(self):
        self.events.append("cpu_offload")

    def unload_ip_adapter(self):
        self.events.append("unload_adapter")
        self.unload_calls += 1
        if self.fail_unload:
            raise RuntimeError("private adapter unload detail")

    def maybe_free_model_hooks(self):
        self.hooks_freed = True

    def remove_all_hooks(self):
        self.hooks_removed = True


def test_reference_adapter_disables_slicing_loads_then_offloads_and_cleans():
    manager = ReferenceAdapterManager()
    pipe = FakePipe(sliced=True)

    with manager.feature_session(pipe, 0.7) as active:
        assert active is pipe
        assert manager.state is AdapterLoadState.LOADED
        assert pipe.scale == 0.7
        assert pipe.load_kwargs[0] == ("h94/IP-Adapter",)
        assert pipe.load_kwargs[1]["subfolder"] == "sdxl_models"
        assert pipe.load_kwargs[1]["weight_name"] == "ip-adapter_sdxl_vit-h.safetensors"
        assert pipe.load_kwargs[1]["image_encoder_folder"] == "models/image_encoder"
        assert pipe.events[:4] == [
            "disable_slicing",
            "load_adapter",
            "set_scale",
            "cpu_offload",
        ]

    assert pipe.unload_calls == 1
    assert pipe.image_encoder is None
    assert pipe.feature_extractor is None
    assert pipe.unet.encoder_hid_proj is None
    assert manager.state is AdapterLoadState.NOT_STARTED


def test_incompatible_big_g_encoder_is_rejected_before_generation():
    manager = ReferenceAdapterManager()
    pipe = FakePipe(encoder_projection_dim=1280, adapter_input_dim=1024)
    with pytest.raises(FeatureLoadError, match="projection_dim=1280, expected 1024"):
        manager.load_feature(pipe)
    assert pipe.unload_calls == 1
    assert "cpu_offload" not in pipe.events
    assert pipe.image_encoder is None
    assert manager.state is AdapterLoadState.NOT_STARTED


def test_reference_adapter_cleans_when_generation_raises():
    manager = ReferenceAdapterManager()
    pipe = FakePipe()
    with pytest.raises(RuntimeError, match="generation failed"):
        with manager.feature_session(pipe):
            raise RuntimeError("generation failed")
    assert pipe.unload_calls == 1
    assert pipe.image_encoder is None
    assert manager.is_loaded is False


def test_partial_load_failure_never_calls_unsafe_unload():
    manager = ReferenceAdapterManager()
    pipe = FakePipe(fail_load=True)
    with pytest.raises(ModelLoadError, match="reference-image adapter"):
        manager.load_feature(pipe)
    assert pipe.unload_calls == 0
    assert pipe.image_encoder is None
    assert pipe.feature_extractor is None
    assert pipe.unet.encoder_hid_proj is None
    assert manager.state is AdapterLoadState.NOT_STARTED


def test_sliced_processor_remaining_after_disable_blocks_adapter_load():
    manager = ReferenceAdapterManager()
    pipe = FakePipe(sliced=True, retain_sliced=True)
    with pytest.raises(ModelLoadError, match="attention slicing remained active"):
        manager.load_feature(pipe)
    assert pipe.events == ["disable_slicing"]
    assert pipe.unload_calls == 0
    assert manager.state is AdapterLoadState.NOT_STARTED


def test_loaded_cleanup_failure_is_explicit_but_references_are_dropped():
    manager = ReferenceAdapterManager()
    pipe = FakePipe(fail_unload=True)
    with pytest.raises(FeatureCleanupError):
        with manager.feature_session(pipe):
            pass
    assert pipe.unload_calls == 1
    assert pipe.image_encoder is None
    assert pipe.feature_extractor is None
    assert manager.state is AdapterLoadState.NOT_STARTED


def test_release_pipeline_frees_diffusers_hooks_and_strong_references():
    manager = ReferenceAdapterManager()
    pipe = FakePipe()
    manager.load_feature(pipe)
    manager.unload_feature()
    manager.release_pipeline(pipe)
    assert pipe.hooks_freed is True
    assert pipe.hooks_removed is True
    assert pipe.image_encoder is None
    assert pipe.feature_extractor is None
    assert manager.is_loaded is False


def test_abnormal_cuda_allocation_blocks_clean_full_reload(monkeypatch):
    manager = ReferenceAdapterManager()
    monkeypatch.setattr("app.core.ip_adapter.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("app.core.ip_adapter.torch.cuda.memory_allocated", lambda: 2 * 1024**3)
    monkeypatch.setattr("app.core.ip_adapter.torch.cuda.memory_reserved", lambda: 3 * 1024**3)
    manager.record_post_cleanup_memory()
    assert manager.last_cleanup_allocated == 2 * 1024**3
    assert manager.last_cleanup_reserved == 3 * 1024**3
    assert manager.cleanup_allows_full_reload is False
