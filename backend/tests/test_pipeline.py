"""Model-free tests for final Full Redesign and Targeted Inpainting wiring."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from app.core import pipeline
from app.core.errors import (
    CudaRequiredError,
    GenerationOutOfMemoryError,
    MaskValidationError,
    ParameterValidationError,
)
from app.core.final_config import FULL_REDESIGN_CONFIG, TARGETED_INPAINT_CONFIG
from app.core.model_manager import model_manager


@pytest.fixture(autouse=True)
def allow_mocked_generation_without_cuda(monkeypatch):
    monkeypatch.setattr(pipeline, "require_cuda", lambda: None)


def _valid_image(size=(512, 512)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[: size[1] // 2] = 40
    arr[size[1] // 2 :] = 200
    return Image.fromarray(arr)


class FakeMidas:
    def __call__(self, image):
        return image


class FakePipe:
    def __init__(self, color=(1, 2, 3)):
        self.calls: list[dict] = []
        self.color = color

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image_arg = kwargs["image"]
        size = image_arg[0].size if isinstance(image_arg, list) else image_arg.size
        return SimpleNamespace(images=[Image.new("RGB", size, self.color)])


class WordTokenizer:
    model_max_length = 77

    def __call__(self, text, **kwargs):
        words = text.replace(",", " ").replace(";", " ").split()
        return {"input_ids": [0] * (len(words) + 2)}


def _install_full_fakes(monkeypatch, pipe_obj=None):
    fake_pipe = pipe_obj or FakePipe()
    monkeypatch.setattr(model_manager, "get_pipe", lambda: fake_pipe)
    monkeypatch.setattr(model_manager, "get_midas", lambda: FakeMidas())
    return fake_pipe


def _install_inpaint_fakes(monkeypatch, pipe_obj=None):
    fake_pipe = pipe_obj or FakePipe((200, 150, 90))
    monkeypatch.setattr(model_manager, "get_inpaint_pipe", lambda: fake_pipe)
    monkeypatch.setattr(model_manager, "get_midas", lambda: FakeMidas())
    return fake_pipe


def test_full_defaults_match_final_config(monkeypatch):
    fake_pipe = _install_full_fakes(monkeypatch)
    result = pipeline.generate_room_result(_valid_image(), style_key="modern")
    call = fake_pipe.calls[0]

    assert call["controlnet_conditioning_scale"] == [0.50, 0.20]
    assert call["num_inference_steps"] == 30
    assert call["guidance_scale"] == 7.5
    assert call["generator"].initial_seed() == 42
    assert result.metadata.seed == FULL_REDESIGN_CONFIG.seed
    assert result.metadata.mode == "full_redesign"
    assert result.metadata.runtime_sec >= 0
    assert result.metadata.peak_vram_gb == 0.0


def test_full_explicit_overrides_are_passed_through(monkeypatch):
    fake_pipe = _install_full_fakes(monkeypatch)
    result = pipeline.generate_room_result(
        _valid_image(),
        depth_scale=0.7,
        canny_scale=0.35,
        num_inference_steps=15,
        guidance_scale=5.0,
        seed=99,
    )
    call = fake_pipe.calls[0]
    assert call["controlnet_conditioning_scale"] == [0.7, 0.35]
    assert call["num_inference_steps"] == 15
    assert call["guidance_scale"] == 5.0
    assert call["generator"].initial_seed() == 99
    assert result.metadata.seed == 99


def test_full_metadata_records_the_token_budgeted_prompt_sent_to_model(monkeypatch):
    fake_pipe = FakePipe()
    fake_pipe.tokenizer = WordTokenizer()
    fake_pipe.tokenizer_2 = WordTokenizer()
    _install_full_fakes(monkeypatch, fake_pipe)
    long_request = " ".join(f"detail{index}" for index in range(100))

    result = pipeline.generate_room_result(
        _valid_image(),
        style_key="modern",
        user_text=long_request,
    )

    actual_prompt = fake_pipe.calls[0]["prompt"]
    assert result.metadata.prompt == actual_prompt
    assert len(fake_pipe.tokenizer(actual_prompt)["input_ids"]) <= 77
    assert "architectural shell" in actual_prompt
    assert "modern minimalist style" in actual_prompt
    assert "detail99" not in actual_prompt


def test_full_pipe_override_bypasses_manager_pipe(monkeypatch):
    default_pipe = _install_full_fakes(monkeypatch)
    override_pipe = FakePipe()
    pipeline.generate_room_result(_valid_image(), pipe_override=override_pipe)
    assert len(override_pipe.calls) == 1
    assert len(default_pipe.calls) == 0


def test_legacy_full_wrapper_still_returns_tuple(monkeypatch):
    _install_full_fakes(monkeypatch)
    image, status = pipeline.redesign_room(_valid_image())
    assert image is not None
    assert status == "ok"


def test_invalid_image_short_circuits_before_cuda_or_models(monkeypatch):
    get_pipe_spy = MagicMock(side_effect=AssertionError("model should not load"))
    monkeypatch.setattr(model_manager, "get_pipe", get_pipe_spy)
    result, message = pipeline.redesign_room(Image.new("RGB", (100, 100), "gray"))
    assert result is None
    assert "small" in message.lower()
    get_pipe_spy.assert_not_called()


def test_inpaint_defaults_strength_and_separate_prompt(monkeypatch):
    fake_pipe = _install_inpaint_fakes(monkeypatch)
    image = _valid_image()
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (0, 0, image.width // 2, image.height))

    result = pipeline.generate_region_result(
        image,
        mask,
        style_key="modern",
        user_text="new headboard",
        target_description="bed",
    )
    call = fake_pipe.calls[0]
    assert call["controlnet_conditioning_scale"] == [0.40, 0.10]
    assert call["strength"] == 1.0
    assert call["num_inference_steps"] == 30
    assert call["guidance_scale"] == 7.5
    assert call["generator"].initial_seed() == 42
    assert "edit only the masked bed" in call["prompt"]
    assert "full interior redesign" not in call["prompt"].lower()
    assert result.metadata.seed == TARGETED_INPAINT_CONFIG.seed
    assert result.metadata.outside_mask_mae == 0.0
    assert result.metadata.mask_coverage == 0.5


def test_inpaint_b1_preserves_outside_and_generated_inside(monkeypatch):
    _install_inpaint_fakes(monkeypatch, FakePipe((200, 150, 90)))
    image = _valid_image()
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (0, 0, image.width // 2, image.height))

    result = pipeline.generate_region_result(image, mask)
    final = np.asarray(result.image)
    model_original = np.asarray(pipeline.resize_for_model(image))
    assert np.array_equal(final[:, final.shape[1] // 2 :], model_original[:, model_original.shape[1] // 2 :])
    assert np.all(final[:, : final.shape[1] // 2] == np.array([200, 150, 90]))


def test_missing_and_empty_masks_fail_before_model_access(monkeypatch):
    get_inpaint_spy = MagicMock(side_effect=AssertionError("model should not load"))
    monkeypatch.setattr(model_manager, "get_inpaint_pipe", get_inpaint_spy)
    with pytest.raises(MaskValidationError):
        pipeline.generate_region_result(_valid_image(), None)
    with pytest.raises(MaskValidationError):
        pipeline.generate_region_result(_valid_image(), Image.new("L", (512, 512), 0))
    get_inpaint_spy.assert_not_called()


def test_invalid_numeric_parameters_fail_before_model_access(monkeypatch):
    get_pipe_spy = MagicMock(side_effect=AssertionError("model should not load"))
    monkeypatch.setattr(model_manager, "get_pipe", get_pipe_spy)
    with pytest.raises(ParameterValidationError):
        pipeline.generate_room_result(_valid_image(), depth_scale=1.5)
    get_pipe_spy.assert_not_called()


def test_cuda_required_failure_is_clean_and_precedes_model_loading(monkeypatch):
    get_pipe_spy = MagicMock(side_effect=AssertionError("model should not load"))
    monkeypatch.setattr(model_manager, "get_pipe", get_pipe_spy)
    monkeypatch.setattr(pipeline, "require_cuda", lambda: (_ for _ in ()).throw(CudaRequiredError()))
    with pytest.raises(CudaRequiredError, match="GPU/CUDA is required"):
        pipeline.generate_room_result(_valid_image())
    get_pipe_spy.assert_not_called()


def test_cuda_oom_is_wrapped_in_domain_safe_error(monkeypatch):
    class OomPipe:
        def __call__(self, **kwargs):
            raise torch.cuda.OutOfMemoryError("raw allocator detail")

    _install_full_fakes(monkeypatch, OomPipe())
    with pytest.raises(GenerationOutOfMemoryError, match="GPU ran out of memory"):
        pipeline.generate_room_result(_valid_image())


def test_redesign_region_retains_no_pipe_override_parameter():
    assert "pipe_override" not in inspect.signature(pipeline.redesign_region).parameters


def test_all_three_generation_entrypoints_remain_callable():
    assert callable(pipeline.generate_room_result)
    assert callable(pipeline.generate_region_result)
    assert callable(pipeline.generate_reference_result)
