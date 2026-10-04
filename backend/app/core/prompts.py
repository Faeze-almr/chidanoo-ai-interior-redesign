"""Final prompt policies for full redesign, targeted editing, and reference mode."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional


CLIP_CONTEXT_TOKENS = 77


# These fragments describe redesignable appearance only. Architecture
# protection is supplied independently and remains active for every style.
STYLE_PROMPTS = {
    "modern": "modern minimalist style, clean lines, neutral palette, sleek new furniture",
    "scandinavian": "scandinavian style, light wood tones, cozy textiles, soft natural light",
    "industrial": "industrial loft style, exposed brick finishes, black metal fixtures, new leather furniture",
    "bohemian": "bohemian style, warm earthy tones, rattan furniture, layered textiles and plants",
    "classic": "classic elegant style, warm wood tones, refined furniture and balanced composition",
    "japandi": "japandi style, warm oak, low-profile furniture, linen, wabi-sabi restraint and soft indirect light",
    "luxury": "luxury style, premium materials, elegant lighting, refined gold accents and sophisticated furniture",
}

ARCHITECTURE_PREFIX = (
    "photorealistic room redesign, exact camera viewpoint, preserve the "
    "architectural shell, room shape, walls and corners; preserve every window "
    "and door position, count, size, shape and perspective; frames, curtains "
    "and material appearance may change"
)

FULL_REDESIGN_CORE = "replace movable furniture and decor"

FULL_QUALITY_SUFFIX = "realistic proportions, materials and lighting"

# Reference mode uses a stronger redesign policy while preserving the
# non-removable architecture constraints used by the final modular pipeline.
REFERENCE_REDESIGN_CORE = (
    "a fully renovated real interior room with completely new furniture and decor"
)
REFERENCE_FALLBACK = "well-designed interior with new furniture"
REFERENCE_QUALITY_SUFFIX = (
    "full interior redesign, photorealistic, high quality, interior design magazine photo"
)

FULL_NEGATIVE_PROMPT = (
    "moved window, resized window, missing window, extra window, moved door, "
    "missing door, changed wall geometry, changed room shape, altered camera "
    "viewpoint, warped perspective, structural renovation, distorted "
    "architecture, same furniture, unchanged furniture, blurry, low quality, "
    "watermark, text, cartoon"
)

# Backward-compatible import name used by existing callers.
NEGATIVE_PROMPT = FULL_NEGATIVE_PROMPT


def _token_count(text: str, tokenizer) -> int:
    """Counts tokens without asking a tokenizer to truncate the prompt."""

    kwargs = {
        "add_special_tokens": True,
        "truncation": False,
        "verbose": False,
    }
    try:
        encoded = tokenizer(text, **kwargs)
    except TypeError:
        kwargs.pop("verbose")
        encoded = tokenizer(text, **kwargs)

    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        input_ids = input_ids[0]
    return len(input_ids)


def _token_limit(tokenizer) -> int:
    model_limit = getattr(tokenizer, "model_max_length", CLIP_CONTEXT_TOKENS)
    try:
        model_limit = int(model_limit)
    except (TypeError, ValueError, OverflowError):
        model_limit = CLIP_CONTEXT_TOKENS
    if model_limit <= 0:
        model_limit = CLIP_CONTEXT_TOKENS
    return min(model_limit, CLIP_CONTEXT_TOKENS)


def prompt_fits_tokenizers(prompt: str, tokenizers: Iterable[object]) -> bool:
    """Returns whether ``prompt`` fits both SDXL CLIP contexts."""

    return all(_token_count(prompt, tokenizer) <= _token_limit(tokenizer) for tokenizer in tokenizers)


def require_prompt_fits_tokenizers(prompt: str, tokenizers: Iterable[object]) -> str:
    """Rejects a prompt that would otherwise be silently truncated by CLIP."""

    tokenizer_list = tuple(tokenizers)
    if tokenizer_list and not prompt_fits_tokenizers(prompt, tokenizer_list):
        raise ValueError("The protected prompt exceeds the SDXL CLIP token budget.")
    return prompt


def _join_prompt(parts: Iterable[str]) -> str:
    return ", ".join(part for part in parts if part)


def build_prompt(
    style_key: Optional[str] = None,
    user_text: Optional[str] = None,
    *,
    tokenizers: Iterable[object] = (),
) -> str:
    """Builds a full-redesign prompt with non-removable architecture rules.

    Architecture, style, and furniture replacement are mandatory. When both
    SDXL tokenizers are available, optional quality wording is removed first
    and only the user-controlled fragment is shortened to stay within each
    77-token CLIP context.
    """

    mandatory_parts = [ARCHITECTURE_PREFIX]
    if style_key and style_key in STYLE_PROMPTS:
        mandatory_parts.append(STYLE_PROMPTS[style_key])
    mandatory_parts.append(FULL_REDESIGN_CORE)

    cleaned_user_text = user_text.strip() if user_text and user_text.strip() else ""
    tokenizer_list = tuple(tokenizers)
    candidate = _join_prompt([*mandatory_parts, cleaned_user_text, FULL_QUALITY_SUFFIX])
    if not tokenizer_list or prompt_fits_tokenizers(candidate, tokenizer_list):
        return candidate

    # Decorative quality wording has the lowest priority.
    candidate = _join_prompt([*mandatory_parts, cleaned_user_text])
    if prompt_fits_tokenizers(candidate, tokenizer_list):
        return candidate

    mandatory_prompt = _join_prompt(mandatory_parts)
    require_prompt_fits_tokenizers(mandatory_prompt, tokenizer_list)
    if not cleaned_user_text:
        return mandatory_prompt

    # Preserve the longest whole-word prefix of user text that fits both
    # tokenizers. Protected and style components are never truncated.
    words = cleaned_user_text.split()
    low, high = 0, len(words)
    while low < high:
        midpoint = (low + high + 1) // 2
        shortened = " ".join(words[:midpoint])
        if prompt_fits_tokenizers(_join_prompt([*mandatory_parts, shortened]), tokenizer_list):
            low = midpoint
        else:
            high = midpoint - 1
    return _join_prompt([*mandatory_parts, " ".join(words[:low])])


def build_reference_prompt(
    style_key: Optional[str] = None,
    user_text: Optional[str] = None,
    *,
    tokenizers: Iterable[object] = (),
) -> str:
    """Builds the accepted final-notebook Reference prompt safely.

    Notebook user text extends the strong redesign prompt; it never replaces
    it. The modular B3.1 architecture policy is also non-removable. Optional
    quality wording is dropped first, then only the user-controlled fragment
    may be shortened to fit both SDXL CLIP tokenizers.
    """

    cleaned_user_text = user_text.strip() if user_text and user_text.strip() else ""
    mandatory_parts = [ARCHITECTURE_PREFIX, REFERENCE_REDESIGN_CORE]
    if style_key and style_key in STYLE_PROMPTS:
        mandatory_parts.append(STYLE_PROMPTS[style_key])
    elif not cleaned_user_text:
        mandatory_parts.append(REFERENCE_FALLBACK)

    tokenizer_list = tuple(tokenizers)
    candidate = _join_prompt(
        [*mandatory_parts, cleaned_user_text, REFERENCE_QUALITY_SUFFIX]
    )
    if not tokenizer_list or prompt_fits_tokenizers(candidate, tokenizer_list):
        return candidate

    candidate = _join_prompt([*mandatory_parts, cleaned_user_text])
    if prompt_fits_tokenizers(candidate, tokenizer_list):
        return candidate

    mandatory_prompt = _join_prompt(mandatory_parts)
    require_prompt_fits_tokenizers(mandatory_prompt, tokenizer_list)
    if not cleaned_user_text:
        return mandatory_prompt

    words = cleaned_user_text.split()
    low, high = 0, len(words)
    while low < high:
        midpoint = (low + high + 1) // 2
        shortened = " ".join(words[:midpoint])
        if prompt_fits_tokenizers(
            _join_prompt([*mandatory_parts, shortened]), tokenizer_list
        ):
            low = midpoint
        else:
            high = midpoint - 1
    return _join_prompt([*mandatory_parts, " ".join(words[:low])])


def _clean_target(target_description: Optional[str]) -> str:
    target = (target_description or "masked target").strip()
    return target or "masked target"


def build_inpaint_prompt(
    target_description: Optional[str] = None,
    user_request: Optional[str] = None,
    style_key: Optional[str] = None,
) -> str:
    """Builds a mask-scoped prompt separate from the full-redesign policy."""

    target = _clean_target(target_description)
    parts = [
        "photorealistic interior",
        f"edit only the masked {target}",
        f"replace only the masked {target} with a visibly different design",
    ]
    if style_key and style_key in STYLE_PROMPTS:
        parts.append(STYLE_PROMPTS[style_key])
    if user_request and user_request.strip():
        parts.append(user_request.strip())
    parts.extend(
        [
            "preserve every unmasked object",
            "preserve the walls, windows, doors, architectural shell and camera view",
            "realistic proportions",
            "sharp details",
        ]
    )
    return ", ".join(parts)


def build_inpaint_negative_prompt(target_description: Optional[str] = None) -> str:
    """Builds target-aware negatives without full-room redesign language."""

    target = _clean_target(target_description)
    return (
        f"same {target}, unchanged {target}, only recolored {target}, duplicate "
        f"{target}, extra {target}, warped {target}, artifacts, changed window, "
        "changed wall, blurry, low quality, watermark, text, cartoon"
    )
