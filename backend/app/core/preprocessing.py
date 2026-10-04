"""Image, control-map, and binary-mask preprocessing."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter

from app.core.errors import InputValidationError, MaskValidationError
from app.core.final_config import (
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    INPUT_LIMITS,
    RESIZE_CONFIG,
)


def check_image_quality(image: Image.Image | None) -> Tuple[bool, str]:
    """Validates basic quality plus defensive size/aspect constraints."""

    if image is None or not isinstance(image, Image.Image):
        return False, "Please provide a readable input image."

    w, h = image.size
    if w <= 0 or h <= 0:
        return False, "The image dimensions are invalid."
    if min(w, h) < INPUT_LIMITS.min_short_side:
        return False, "Image is too small (short side must be at least 256px)."
    if max(w, h) > INPUT_LIMITS.max_dimension or w * h > INPUT_LIMITS.max_pixels:
        return False, "Image dimensions are too large; please use a smaller image."
    if max(w, h) / min(w, h) > INPUT_LIMITS.max_aspect_ratio:
        return False, "Image aspect ratio is too extreme for this generation pipeline."

    gray = np.asarray(image.convert("L"))
    if gray.mean() < 25:
        return False, "Image is too dark; please upload a photo with better lighting."
    if gray.std() < 10:
        return False, "Image has too little contrast; the space's details are hard to make out."

    return True, "ok"


def require_valid_image(image: Image.Image | None) -> Image.Image:
    """Returns a validated PIL image or raises a domain-safe error."""

    ok, message = check_image_quality(image)
    if not ok:
        raise InputValidationError(message)
    assert image is not None
    return image


def resize_for_model(
    image: Image.Image,
    target_size: int = RESIZE_CONFIG.target_short_side,
    multiple: int = RESIZE_CONFIG.multiple,
) -> Image.Image:
    """Scales the short side and rounds both dimensions to ``multiple``."""

    if target_size <= 0 or multiple <= 0:
        raise InputValidationError("Resize settings must be positive values.")
    image = image.convert("RGB")
    w, h = image.size
    scale = target_size / min(w, h)
    new_w = max(multiple, int(round(w * scale / multiple) * multiple))
    new_h = max(multiple, int(round(h * scale / multiple) * multiple))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def get_depth_map(image: Image.Image, midas) -> Image.Image:
    """Builds the MiDaS depth conditioning image."""

    with torch.no_grad():
        depth = midas(image)
    return depth.resize(image.size)


def get_canny_map(
    image: Image.Image,
    low: int = CANNY_LOW_THRESHOLD,
    high: int = CANNY_HIGH_THRESHOLD,
) -> Image.Image:
    """Builds the three-channel Canny conditioning image."""

    arr = np.asarray(image)
    edges = cv2.Canny(arr, low, high)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)


def normalize_mask(mask: Image.Image | None, threshold: int = 128) -> Image.Image:
    """Converts a supplied mask to strict binary L mode (white is editable)."""

    if mask is None or not isinstance(mask, Image.Image):
        raise MaskValidationError("Please provide an edit mask.")
    if not 0 <= threshold <= 255:
        raise MaskValidationError("Mask threshold must be between 0 and 255.")
    grayscale = mask.convert("L")
    return grayscale.point(lambda value: 255 if value >= threshold else 0, mode="L")


def mask_coverage(mask: Image.Image) -> float:
    """Returns the editable fraction of a binary-normalized mask."""

    binary = normalize_mask(mask)
    pixels = np.asarray(binary, dtype=np.uint8)
    return float(np.count_nonzero(pixels) / pixels.size)


def validate_mask(
    mask: Image.Image | None,
    min_coverage: float = INPUT_LIMITS.min_mask_coverage,
) -> float:
    """Validates mask presence and meaningful editable coverage."""

    binary = normalize_mask(mask)
    coverage = mask_coverage(binary)
    if coverage == 0.0:
        raise MaskValidationError("The edit mask is empty; paint the area you want to change.")
    if coverage < min_coverage:
        percent = min_coverage * 100
        raise MaskValidationError(
            f"The edit mask is too small; cover at least {percent:.1f}% of the image."
        )
    return coverage


def validate_mask_alignment(
    mask: Image.Image,
    source_image: Image.Image,
    tolerance: float = INPUT_LIMITS.mask_aspect_ratio_tolerance,
) -> None:
    """Rejects masks whose aspect ratio cannot align safely with the source."""

    mask_w, mask_h = mask.size
    image_w, image_h = source_image.size
    if min(mask_w, mask_h, image_w, image_h) <= 0:
        raise MaskValidationError("Mask dimensions are invalid.")
    mask_ratio = mask_w / mask_h
    image_ratio = image_w / image_h
    relative_difference = abs(mask_ratio - image_ratio) / image_ratio
    if relative_difference > tolerance:
        raise MaskValidationError(
            "Mask aspect ratio does not match the source image; create the mask on the source image."
        )


def resize_mask(mask: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resizes a binary mask using nearest-neighbor interpolation only."""

    binary = normalize_mask(mask)
    resized = binary.resize(target_size, Image.Resampling.NEAREST)
    return normalize_mask(resized)


def prepare_mask(
    mask: Image.Image | None,
    source_image: Image.Image,
    model_image: Image.Image,
) -> tuple[Image.Image, float]:
    """Normalizes, validates, aligns, and model-resizes an edit mask."""

    binary = normalize_mask(mask)
    validate_mask_alignment(binary, source_image)
    validate_mask(binary)
    resized = resize_mask(binary, model_image.size)
    coverage = validate_mask(resized)
    return resized, coverage


def resize_mask_to_image(mask: Image.Image, image: Image.Image) -> Image.Image:
    """Backward-compatible binary nearest-neighbor mask resize helper."""

    return resize_mask(mask, image.size)


def feather_mask(mask: Image.Image, radius: float = 4.0) -> Image.Image:
    """Applies inward-only feathering while keeping all outside pixels black."""

    binary = normalize_mask(mask)
    if radius <= 0:
        return binary
    blurred = binary.filter(ImageFilter.GaussianBlur(radius=radius))
    return ImageChops.multiply(binary, blurred)


def restore_outside_mask(
    original_model_image: Image.Image,
    generated_image: Image.Image,
    binary_mask: Image.Image,
) -> Image.Image:
    """Applies exact outside-mask restoration in model-image coordinates."""

    original = original_model_image.convert("RGB")
    generated = generated_image.convert("RGB")
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    mask = resize_mask(binary_mask, original.size)
    return Image.composite(generated, original, mask)


def composite_masked_result(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Backward-compatible alias for exact outside-mask restoration."""

    return restore_outside_mask(original, generated, mask)


def compute_outside_mask_mae(
    original_model_image: Image.Image,
    final_image: Image.Image,
    binary_mask: Image.Image,
) -> float:
    """Computes mean absolute error over protected (black-mask) pixels."""

    original = np.asarray(original_model_image.convert("RGB"), dtype=np.float32)
    final = np.asarray(final_image.convert("RGB"), dtype=np.float32)
    if original.shape != final.shape:
        raise MaskValidationError("Images must have identical dimensions for mask validation.")
    mask = np.asarray(resize_mask(binary_mask, original_model_image.size), dtype=np.uint8)
    outside = mask == 0
    if not np.any(outside):
        return 0.0
    return float(np.abs(original - final)[outside].mean())

