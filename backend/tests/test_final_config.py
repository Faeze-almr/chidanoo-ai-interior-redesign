from app.core.final_config import (
    FULL_REDESIGN_CONFIG,
    MODEL_CONFIG,
    RESIZE_CONFIG,
    TARGETED_INPAINT_CONFIG,
)


def test_final_full_redesign_configuration():
    assert FULL_REDESIGN_CONFIG.seed == 42
    assert FULL_REDESIGN_CONFIG.steps == 30
    assert FULL_REDESIGN_CONFIG.guidance_scale == 7.5
    assert FULL_REDESIGN_CONFIG.depth_scale == 0.50
    assert FULL_REDESIGN_CONFIG.canny_scale == 0.20


def test_final_targeted_inpaint_configuration():
    assert TARGETED_INPAINT_CONFIG.seed == 42
    assert TARGETED_INPAINT_CONFIG.steps == 30
    assert TARGETED_INPAINT_CONFIG.guidance_scale == 7.5
    assert TARGETED_INPAINT_CONFIG.depth_scale == 0.40
    assert TARGETED_INPAINT_CONFIG.canny_scale == 0.10
    assert TARGETED_INPAINT_CONFIG.strength == 1.0


def test_resize_and_model_configuration():
    assert RESIZE_CONFIG.target_short_side == 1024
    assert RESIZE_CONFIG.multiple == 8
    assert MODEL_CONFIG.base_model_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert MODEL_CONFIG.scheduler == "DPMSolverMultistepScheduler"
    assert MODEL_CONFIG.use_karras_sigmas is True

