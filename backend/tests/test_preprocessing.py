import numpy as np
import pytest
from PIL import Image

from app.core.errors import MaskValidationError
from app.core.preprocessing import (
    check_image_quality,
    compute_outside_mask_mae,
    feather_mask,
    get_canny_map,
    mask_coverage,
    normalize_mask,
    resize_for_model,
    resize_mask,
    restore_outside_mask,
    validate_mask,
    validate_mask_alignment,
)


def _solid_image(size, color, mode="RGB"):
    return Image.new(mode, size, color)


def _valid_image(size=(512, 512)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[: size[1] // 2] = 40
    arr[size[1] // 2 :] = 200
    return Image.fromarray(arr)


def test_rejects_missing_image():
    ok, message = check_image_quality(None)
    assert ok is False
    assert "provide" in message.lower()


def test_rejects_image_below_min_short_side():
    ok, message = check_image_quality(_solid_image((300, 200), (128, 128, 128)))
    assert ok is False
    assert "small" in message.lower()


def test_rejects_too_dark_and_low_contrast_images():
    assert check_image_quality(_solid_image((512, 512), (5, 5, 5)))[0] is False
    ok, message = check_image_quality(_solid_image((512, 512), (128, 128, 128)))
    assert ok is False
    assert "contrast" in message.lower()


def test_rejects_extreme_dimensions_and_aspect_ratio():
    ok, message = check_image_quality(_valid_image((9000, 512)))
    assert ok is False
    assert "large" in message.lower() or "aspect" in message.lower()
    ok, message = check_image_quality(_valid_image((4096, 512)))
    assert ok is False
    assert "aspect" in message.lower()


def test_accepts_normal_image():
    assert check_image_quality(_valid_image()) == (True, "ok")


def test_resize_rounds_to_multiple_of_eight_and_rgb():
    resized = resize_for_model(_solid_image((777, 501), 200, mode="L"))
    assert resized.mode == "RGB"
    assert resized.width % 8 == 0
    assert resized.height % 8 == 0
    assert min(resized.size) in range(1016, 1033)


def test_get_canny_map_returns_same_size_rgb_edges():
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[64:192, 64:192] = 255
    edges = get_canny_map(Image.fromarray(arr))
    assert edges.size == (256, 256)
    assert edges.mode == "RGB"
    assert np.asarray(edges).max() > 0


def test_missing_mask_is_rejected():
    with pytest.raises(MaskValidationError, match="provide an edit mask"):
        normalize_mask(None)


def test_empty_and_near_zero_masks_are_rejected():
    with pytest.raises(MaskValidationError, match="empty"):
        validate_mask(Image.new("L", (100, 100), 0))

    one_pixel = Image.new("L", (100, 100), 0)
    one_pixel.putpixel((0, 0), 255)
    with pytest.raises(MaskValidationError, match="too small"):
        validate_mask(one_pixel)


def test_mask_normalization_is_strictly_binary():
    mask = Image.fromarray(np.array([[0, 127], [128, 255]], dtype=np.uint8), mode="L")
    normalized = np.asarray(normalize_mask(mask))
    assert set(np.unique(normalized)) == {0, 255}
    assert normalized.tolist() == [[0, 0], [255, 255]]


def test_mask_resize_uses_nearest_neighbor_and_stays_binary():
    mask = Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8), mode="L")
    resized = np.asarray(resize_mask(mask, (4, 4)))
    assert set(np.unique(resized)) == {0, 255}
    assert np.all(resized[:2, :2] == 0)
    assert np.all(resized[:2, 2:] == 255)


def test_mask_aspect_mismatch_is_rejected():
    with pytest.raises(MaskValidationError, match="aspect ratio"):
        validate_mask_alignment(
            Image.new("L", (100, 100), 255),
            Image.new("RGB", (200, 100), "white"),
        )


def test_mask_coverage_reports_editable_fraction():
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (0, 0, 50, 100))
    assert mask_coverage(mask) == 0.5


def test_inward_feather_never_modifies_outside_boundary():
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (50, 0, 100, 100))
    feathered = np.asarray(feather_mask(mask, radius=4))
    assert np.all(feathered[:, :50] == 0)
    assert np.any((feathered[:, 50:] > 0) & (feathered[:, 50:] < 255))


def test_b1_restoration_preserves_outside_and_generated_inside():
    original = Image.new("RGB", (100, 100), (10, 20, 30))
    generated = Image.new("RGB", (100, 100), (200, 150, 90))
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (0, 0, 50, 100))

    final = restore_outside_mask(original, generated, mask)
    pixels = np.asarray(final)
    assert tuple(pixels[10, 90]) == (10, 20, 30)
    assert tuple(pixels[10, 10]) == (200, 150, 90)
    assert compute_outside_mask_mae(original, final, mask) == 0.0

