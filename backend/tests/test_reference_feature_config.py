from app.core.final_config import (
    FULL_REDESIGN_CONFIG,
    REFERENCE_FEATURE_CONFIG,
    REFERENCE_REDESIGN_CONFIG,
    TARGETED_INPAINT_CONFIG,
)
from app.core.prompts import FULL_NEGATIVE_PROMPT, STYLE_PROMPTS


def test_reference_feature_configuration():
    assert REFERENCE_FEATURE_CONFIG.adapter_id == "h94/IP-Adapter"
    assert REFERENCE_FEATURE_CONFIG.subfolder == "sdxl_models"
    assert REFERENCE_FEATURE_CONFIG.weight_name == "ip-adapter_sdxl_vit-h.safetensors"
    assert REFERENCE_FEATURE_CONFIG.image_encoder_folder == "models/image_encoder"
    assert REFERENCE_FEATURE_CONFIG.expected_projection_dim == 1024
    assert REFERENCE_FEATURE_CONFIG.default_scale == 0.60
    assert REFERENCE_REDESIGN_CONFIG == FULL_REDESIGN_CONFIG


def test_frozen_configuration_is_unchanged_after_reference_integration():
    assert (
        FULL_REDESIGN_CONFIG.seed,
        FULL_REDESIGN_CONFIG.steps,
        FULL_REDESIGN_CONFIG.guidance_scale,
        FULL_REDESIGN_CONFIG.depth_scale,
        FULL_REDESIGN_CONFIG.canny_scale,
    ) == (42, 30, 7.5, 0.50, 0.20)
    assert (
        TARGETED_INPAINT_CONFIG.seed,
        TARGETED_INPAINT_CONFIG.steps,
        TARGETED_INPAINT_CONFIG.guidance_scale,
        TARGETED_INPAINT_CONFIG.depth_scale,
        TARGETED_INPAINT_CONFIG.canny_scale,
        TARGETED_INPAINT_CONFIG.strength,
    ) == (42, 30, 7.5, 0.40, 0.10, 1.0)
    assert "identical layout" not in FULL_NEGATIVE_PROMPT.lower()
    assert {"japandi", "luxury"} <= set(STYLE_PROMPTS)
