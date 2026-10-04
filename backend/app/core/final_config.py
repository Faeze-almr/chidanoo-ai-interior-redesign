"""Frozen canonical inference configuration.

Keeping these values in one module prevents API routes, pipeline functions,
tests, and future thin UIs from developing independent defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    seed: int
    steps: int
    guidance_scale: float
    depth_scale: float
    canny_scale: float
    strength: float | None = None


@dataclass(frozen=True)
class ResizeConfig:
    target_short_side: int
    multiple: int


@dataclass(frozen=True)
class ModelConfig:
    base_model_id: str
    depth_controlnet_id: str
    canny_controlnet_id: str
    vae_id: str
    midas_repo_id: str
    scheduler: str
    use_karras_sigmas: bool


@dataclass(frozen=True)
class ReferenceFeatureConfig:
    adapter_id: str
    subfolder: str
    weight_name: str
    image_encoder_folder: str
    expected_projection_dim: int
    default_scale: float


@dataclass(frozen=True)
class InputLimits:
    min_short_side: int
    max_dimension: int
    max_pixels: int
    max_aspect_ratio: float
    mask_aspect_ratio_tolerance: float
    min_mask_coverage: float


FULL_REDESIGN_CONFIG = GenerationConfig(
    seed=42,
    steps=30,
    guidance_scale=7.5,
    depth_scale=0.50,
    canny_scale=0.20,
)

TARGETED_INPAINT_CONFIG = GenerationConfig(
    seed=42,
    steps=30,
    guidance_scale=7.5,
    depth_scale=0.40,
    canny_scale=0.10,
    strength=1.0,
)

# Reference-image generation intentionally inherits the validated Full
# Redesign settings. The only additional conditioning parameter is the
# IP-Adapter scale.
REFERENCE_REDESIGN_CONFIG = GenerationConfig(
    seed=42,
    steps=30,
    guidance_scale=7.5,
    depth_scale=0.50,
    canny_scale=0.20,
)

RESIZE_CONFIG = ResizeConfig(target_short_side=1024, multiple=8)

MODEL_CONFIG = ModelConfig(
    base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
    depth_controlnet_id="diffusers/controlnet-depth-sdxl-1.0",
    canny_controlnet_id="diffusers/controlnet-canny-sdxl-1.0",
    vae_id="madebyollin/sdxl-vae-fp16-fix",
    midas_repo_id="lllyasviel/Annotators",
    scheduler="DPMSolverMultistepScheduler",
    use_karras_sigmas=True,
)

REFERENCE_FEATURE_CONFIG = ReferenceFeatureConfig(
    adapter_id="h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl_vit-h.safetensors",
    image_encoder_folder="models/image_encoder",
    expected_projection_dim=1024,
    default_scale=0.60,
)

# Defensive product limits prevent pathological uploads from expanding into
# impractical model images while preserving the accepted final input range.
INPUT_LIMITS = InputLimits(
    min_short_side=256,
    max_dimension=8192,
    max_pixels=40_000_000,
    max_aspect_ratio=4.0,
    mask_aspect_ratio_tolerance=0.02,
    min_mask_coverage=0.001,
)

CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200
