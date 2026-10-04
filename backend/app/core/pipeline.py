"""Canonical generation pipelines with safe errors and metadata."""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Tuple

import torch
from PIL import Image

from app.core.errors import (
    FeatureCleanupError,
    GenerationDomainError,
    GenerationOutOfMemoryError,
    GenerationRuntimeError,
    InputValidationError,
    ParameterValidationError,
)
from app.core.final_config import (
    FULL_REDESIGN_CONFIG,
    MODEL_CONFIG,
    REFERENCE_FEATURE_CONFIG,
    REFERENCE_REDESIGN_CONFIG,
    TARGETED_INPAINT_CONFIG,
)
from app.core.ip_adapter import reference_adapter_manager
from app.core.model_manager import inference_lock, model_manager, require_cuda
from app.core.preprocessing import (
    compute_outside_mask_mae,
    get_canny_map,
    get_depth_map,
    mask_coverage,
    normalize_mask,
    require_valid_image,
    resize_for_model,
    resize_mask,
    restore_outside_mask,
    validate_mask,
    validate_mask_alignment,
)
from app.core.prompts import (
    FULL_NEGATIVE_PROMPT,
    build_inpaint_negative_prompt,
    build_inpaint_prompt,
    build_prompt,
    build_reference_prompt,
    require_prompt_fits_tokenizers,
)

logger = logging.getLogger(__name__)


def _serialized_generation(function):
    """Protects the shared SDXL/UNet state for every direct caller."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with inference_lock:
            return function(*args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class GenerationMetadata:
    mode: str
    timestamp: str
    model: str
    seed: int
    steps: int
    cfg: float
    depth_scale: float
    canny_scale: float
    resolution: tuple[int, int]
    runtime_sec: float
    peak_vram_gb: float
    prompt: str
    negative_prompt: str
    status: str
    strength: float | None = None
    mask_coverage: float | None = None
    outside_mask_mae: float | None = None
    reference_scale: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    image: Image.Image
    metadata: GenerationMetadata
    raw_image: Image.Image | None = None


def validate_generation_parameters(
    *,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    depth_scale: float,
    canny_scale: float,
    strength: float | None = None,
    reference_scale: float | None = None,
) -> None:
    """Validates caller-controlled numeric values before model access."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ParameterValidationError("Seed must be an integer between 0 and 2^63-1.")
    if (
        isinstance(num_inference_steps, bool)
        or not isinstance(num_inference_steps, int)
        or not 1 <= num_inference_steps <= 150
    ):
        raise ParameterValidationError("Inference steps must be an integer between 1 and 150.")

    numeric_values = {
        "CFG": guidance_scale,
        "depth scale": depth_scale,
        "Canny scale": canny_scale,
    }
    for label, value in numeric_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ParameterValidationError(f"{label} must be a finite number.")
    if not 0.0 <= guidance_scale <= 30.0:
        raise ParameterValidationError("CFG must be between 0 and 30.")
    if not 0.0 <= depth_scale <= 1.0 or not 0.0 <= canny_scale <= 1.0:
        raise ParameterValidationError("ControlNet scales must be between 0 and 1.")
    if strength is not None:
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise ParameterValidationError("Inpaint strength must be a finite number.")
        if not 0.0 <= strength <= 1.0:
            raise ParameterValidationError("Inpaint strength must be between 0 and 1.")
    if reference_scale is not None:
        if (
            isinstance(reference_scale, bool)
            or not isinstance(reference_scale, (int, float))
            or not math.isfinite(reference_scale)
        ):
            raise ParameterValidationError("Reference scale must be a finite number.")
        if not 0.0 <= reference_scale <= 1.0:
            raise ParameterValidationError("Reference scale must be between 0 and 1.")


def _begin_metrics() -> tuple[str, float]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return datetime.now(timezone.utc).isoformat(), time.perf_counter()


def _finish_metrics(started: float) -> tuple[float, float]:
    runtime = time.perf_counter() - started
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    return runtime, peak_vram


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            try:
                ipc_collect()
            except Exception:
                logger.exception("CUDA IPC cleanup failed")


def _pipeline_tokenizers(pipe) -> tuple[object, ...]:
    """Returns the two SDXL tokenizers when a real pipeline exposes them."""

    return tuple(
        tokenizer
        for tokenizer in (getattr(pipe, "tokenizer", None), getattr(pipe, "tokenizer_2", None))
        if tokenizer is not None
    )


@_serialized_generation
def generate_room_result(
    input_image: Image.Image,
    style_key: Optional[str] = None,
    user_text: Optional[str] = None,
    depth_scale: float = FULL_REDESIGN_CONFIG.depth_scale,
    canny_scale: float = FULL_REDESIGN_CONFIG.canny_scale,
    num_inference_steps: int = FULL_REDESIGN_CONFIG.steps,
    guidance_scale: float = FULL_REDESIGN_CONFIG.guidance_scale,
    seed: int = FULL_REDESIGN_CONFIG.seed,
    pipe_override=None,
) -> GenerationResult:
    """Runs Full Redesign and returns the image plus structured metadata."""

    source = require_valid_image(input_image)
    validate_generation_parameters(
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        depth_scale=depth_scale,
        canny_scale=canny_scale,
    )
    require_cuda()

    # Resolve/load models before timing so first-load time is never reported as
    # generation time.
    active_pipe = pipe_override if pipe_override is not None else model_manager.get_pipe()
    midas = model_manager.get_midas()
    tokenizers = _pipeline_tokenizers(active_pipe)
    prompt = build_prompt(style_key, user_text, tokenizers=tokenizers)
    negative_prompt = require_prompt_fits_tokenizers(FULL_NEGATIVE_PROMPT, tokenizers)

    timestamp, started = _begin_metrics()
    depth_map = None
    canny_map = None
    try:
        image = resize_for_model(source)
        depth_map = get_depth_map(image, midas)
        canny_map = get_canny_map(image)
        generator = torch.Generator(device=model_manager.device).manual_seed(seed)

        generated = active_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=[depth_map, canny_map],
            controlnet_conditioning_scale=[depth_scale, canny_scale],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        runtime_sec, peak_vram_gb = _finish_metrics(started)
    except torch.cuda.OutOfMemoryError as exc:
        logger.exception("CUDA OOM during full redesign")
        raise GenerationOutOfMemoryError() from exc
    except GenerationDomainError:
        raise
    except RuntimeError as exc:
        logger.exception("Runtime failure during full redesign")
        raise GenerationRuntimeError() from exc
    except Exception as exc:
        logger.exception("Unexpected failure during full redesign")
        raise GenerationRuntimeError() from exc
    finally:
        depth_map = None
        canny_map = None
        _cleanup_cuda()

    metadata = GenerationMetadata(
        mode="full_redesign",
        timestamp=timestamp,
        model=MODEL_CONFIG.base_model_id,
        seed=seed,
        steps=num_inference_steps,
        cfg=float(guidance_scale),
        depth_scale=float(depth_scale),
        canny_scale=float(canny_scale),
        resolution=generated.size,
        runtime_sec=round(runtime_sec, 4),
        peak_vram_gb=round(peak_vram_gb, 4),
        prompt=prompt,
        negative_prompt=negative_prompt,
        status="succeeded",
    )
    return GenerationResult(image=generated, metadata=metadata)


def redesign_room(
    input_image: Image.Image,
    style_key: Optional[str] = None,
    user_text: Optional[str] = None,
    depth_scale: float = FULL_REDESIGN_CONFIG.depth_scale,
    canny_scale: float = FULL_REDESIGN_CONFIG.canny_scale,
    num_inference_steps: int = FULL_REDESIGN_CONFIG.steps,
    guidance_scale: float = FULL_REDESIGN_CONFIG.guidance_scale,
    seed: int = FULL_REDESIGN_CONFIG.seed,
    pipe_override=None,
) -> Tuple[Optional[Image.Image], str]:
    """Backward-compatible Full Redesign wrapper returning ``(image, status)``."""

    try:
        result = generate_room_result(
            input_image,
            style_key=style_key,
            user_text=user_text,
            depth_scale=depth_scale,
            canny_scale=canny_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            pipe_override=pipe_override,
        )
    except InputValidationError as exc:
        return None, exc.user_message
    return result.image, "ok"


@_serialized_generation
def generate_reference_result(
    input_image: Image.Image,
    reference_image: Image.Image,
    style_key: Optional[str] = None,
    user_text: Optional[str] = None,
    depth_scale: float = REFERENCE_REDESIGN_CONFIG.depth_scale,
    canny_scale: float = REFERENCE_REDESIGN_CONFIG.canny_scale,
    reference_scale: float = REFERENCE_FEATURE_CONFIG.default_scale,
    num_inference_steps: int = REFERENCE_REDESIGN_CONFIG.steps,
    guidance_scale: float = REFERENCE_REDESIGN_CONFIG.guidance_scale,
    seed: int = REFERENCE_REDESIGN_CONFIG.seed,
) -> GenerationResult:
    """Runs Reference mode in a disposable, low-VRAM pipeline."""

    source = require_valid_image(input_image)
    reference = require_valid_image(reference_image).convert("RGB")
    validate_generation_parameters(
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        depth_scale=depth_scale,
        canny_scale=canny_scale,
        reference_scale=reference_scale,
    )
    require_cuda()

    timestamp = ""
    started = 0.0
    reference_pipe = None
    active_pipe = None
    midas = None
    depth_map = None
    canny_map = None
    try:
        # Release the canonical Full/Inpaint graph before allocating fresh
        # Reference-only weights. The new pipe is never stored by ModelManager.
        model_manager.unload_all()
        reference_pipe = model_manager.create_reference_pipe()
        midas = model_manager.get_midas()
        tokenizers = _pipeline_tokenizers(reference_pipe)
        prompt = build_reference_prompt(style_key, user_text, tokenizers=tokenizers)
        negative_prompt = require_prompt_fits_tokenizers(FULL_NEGATIVE_PROMPT, tokenizers)

        with reference_adapter_manager.feature_session(reference_pipe, reference_scale) as active_pipe:
            timestamp, started = _begin_metrics()
            image = resize_for_model(source)
            depth_map = get_depth_map(image, midas)
            canny_map = get_canny_map(image)
            generator = torch.Generator(device=model_manager.device).manual_seed(seed)
            generated = active_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=[depth_map, canny_map],
                ip_adapter_image=reference,
                controlnet_conditioning_scale=[depth_scale, canny_scale],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            runtime_sec, peak_vram_gb = _finish_metrics(started)
    except FeatureCleanupError:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        logger.exception("CUDA OOM during reference-image redesign")
        raise GenerationOutOfMemoryError() from exc
    except GenerationDomainError:
        raise
    except RuntimeError as exc:
        logger.exception("Runtime failure during reference-image redesign")
        raise GenerationRuntimeError() from exc
    except Exception as exc:
        logger.exception("Unexpected failure during reference-image redesign")
        raise GenerationRuntimeError() from exc
    finally:
        cleanup_pipe = reference_pipe if reference_pipe is not None else active_pipe
        reference_adapter_manager.release_pipeline(cleanup_pipe)
        # Also evict the temporary MiDaS instance. A later Full/Inpaint call
        # must request an entirely clean canonical pipeline and preprocessor.
        model_manager.unload_all()
        depth_map = None
        canny_map = None
        midas = None
        active_pipe = None
        reference_pipe = None
        del cleanup_pipe
        _cleanup_cuda()
        reference_adapter_manager.record_post_cleanup_memory()

    metadata = GenerationMetadata(
        mode="reference_redesign",
        timestamp=timestamp,
        model=MODEL_CONFIG.base_model_id,
        seed=seed,
        steps=num_inference_steps,
        cfg=float(guidance_scale),
        depth_scale=float(depth_scale),
        canny_scale=float(canny_scale),
        reference_scale=float(reference_scale),
        resolution=generated.size,
        runtime_sec=round(runtime_sec, 4),
        peak_vram_gb=round(peak_vram_gb, 4),
        prompt=prompt,
        negative_prompt=negative_prompt,
        status="succeeded",
    )
    return GenerationResult(image=generated, metadata=metadata)


def redesign_with_reference(
    input_image: Image.Image,
    reference_image: Image.Image,
    user_text: Optional[str] = None,
    depth_scale: float = REFERENCE_REDESIGN_CONFIG.depth_scale,
    canny_scale: float = REFERENCE_REDESIGN_CONFIG.canny_scale,
    reference_scale: float = REFERENCE_FEATURE_CONFIG.default_scale,
    num_inference_steps: int = REFERENCE_REDESIGN_CONFIG.steps,
    guidance_scale: float = REFERENCE_REDESIGN_CONFIG.guidance_scale,
    seed: int = REFERENCE_REDESIGN_CONFIG.seed,
    style_key: Optional[str] = None,
) -> Tuple[Optional[Image.Image], str]:
    """Compatibility wrapper for the reference-image API."""

    try:
        result = generate_reference_result(
            input_image,
            reference_image,
            style_key=style_key,
            user_text=user_text,
            depth_scale=depth_scale,
            canny_scale=canny_scale,
            reference_scale=reference_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    except InputValidationError as exc:
        return None, exc.user_message
    return result.image, "ok"


@_serialized_generation
def generate_region_result(
    input_image: Image.Image,
    mask_image: Image.Image,
    style_key: Optional[str] = None,
    user_text: Optional[str] = None,
    depth_scale: float = TARGETED_INPAINT_CONFIG.depth_scale,
    canny_scale: float = TARGETED_INPAINT_CONFIG.canny_scale,
    num_inference_steps: int = TARGETED_INPAINT_CONFIG.steps,
    guidance_scale: float = TARGETED_INPAINT_CONFIG.guidance_scale,
    seed: int = TARGETED_INPAINT_CONFIG.seed,
    target_description: str = "target area",
    strength: float = TARGETED_INPAINT_CONFIG.strength or 1.0,
    include_raw_output: bool = False,
) -> GenerationResult:
    """Runs targeted inpainting followed by mandatory B1 restoration."""

    source = require_valid_image(input_image)
    validate_generation_parameters(
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        depth_scale=depth_scale,
        canny_scale=canny_scale,
        strength=strength,
    )

    # Validate the original-space mask before any model access.
    binary_source_mask = normalize_mask(mask_image)
    validate_mask_alignment(binary_source_mask, source)
    validate_mask(binary_source_mask)
    require_cuda()

    inpaint_pipe = model_manager.get_inpaint_pipe()
    midas = model_manager.get_midas()
    prompt = build_inpaint_prompt(target_description, user_text, style_key)
    negative_prompt = build_inpaint_negative_prompt(target_description)

    timestamp, started = _begin_metrics()
    depth_map = None
    canny_map = None
    raw_output = None
    try:
        image = resize_for_model(source)
        mask = resize_mask(binary_source_mask, image.size)
        coverage = validate_mask(mask)
        depth_map = get_depth_map(image, midas)
        canny_map = get_canny_map(image)
        generator = torch.Generator(device=model_manager.device).manual_seed(seed)

        raw_output = inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            control_image=[depth_map, canny_map],
            controlnet_conditioning_scale=[depth_scale, canny_scale],
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # This is the final pixel operation. Nothing may alter protected pixels
        # after this exact model-space composite.
        protected_output = restore_outside_mask(image, raw_output, mask)
        outside_mae = compute_outside_mask_mae(image, protected_output, mask)
        runtime_sec, peak_vram_gb = _finish_metrics(started)
    except torch.cuda.OutOfMemoryError as exc:
        logger.exception("CUDA OOM during targeted inpainting")
        raise GenerationOutOfMemoryError() from exc
    except GenerationDomainError:
        raise
    except RuntimeError as exc:
        logger.exception("Runtime failure during targeted inpainting")
        raise GenerationRuntimeError() from exc
    except Exception as exc:
        logger.exception("Unexpected failure during targeted inpainting")
        raise GenerationRuntimeError() from exc
    finally:
        depth_map = None
        canny_map = None
        _cleanup_cuda()

    metadata = GenerationMetadata(
        mode="targeted_inpaint",
        timestamp=timestamp,
        model=MODEL_CONFIG.base_model_id,
        seed=seed,
        steps=num_inference_steps,
        cfg=float(guidance_scale),
        depth_scale=float(depth_scale),
        canny_scale=float(canny_scale),
        strength=float(strength),
        resolution=protected_output.size,
        runtime_sec=round(runtime_sec, 4),
        peak_vram_gb=round(peak_vram_gb, 4),
        prompt=prompt,
        negative_prompt=negative_prompt,
        status="succeeded",
        mask_coverage=round(mask_coverage(mask), 6),
        outside_mask_mae=outside_mae,
    )
    return GenerationResult(
        image=protected_output,
        metadata=metadata,
        raw_image=raw_output if include_raw_output else None,
    )


def redesign_region(
    input_image: Image.Image,
    mask_image: Image.Image,
    style_key: Optional[str] = None,
    user_text: Optional[str] = None,
    depth_scale: float = TARGETED_INPAINT_CONFIG.depth_scale,
    canny_scale: float = TARGETED_INPAINT_CONFIG.canny_scale,
    num_inference_steps: int = TARGETED_INPAINT_CONFIG.steps,
    guidance_scale: float = TARGETED_INPAINT_CONFIG.guidance_scale,
    seed: int = TARGETED_INPAINT_CONFIG.seed,
    target_description: str = "target area",
    strength: float = TARGETED_INPAINT_CONFIG.strength or 1.0,
) -> Tuple[Optional[Image.Image], str]:
    """Backward-compatible targeted-edit wrapper returning protected output."""

    try:
        result = generate_region_result(
            input_image,
            mask_image,
            style_key=style_key,
            user_text=user_text,
            depth_scale=depth_scale,
            canny_scale=canny_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            target_description=target_description,
            strength=strength,
        )
    except InputValidationError as exc:
        return None, exc.user_message
    return result.image, "ok"
