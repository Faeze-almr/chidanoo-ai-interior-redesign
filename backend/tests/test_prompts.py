from app.core.prompts import (
    ARCHITECTURE_PREFIX,
    CLIP_CONTEXT_TOKENS,
    FULL_NEGATIVE_PROMPT,
    REFERENCE_FALLBACK,
    REFERENCE_REDESIGN_CORE,
    STYLE_PROMPTS,
    build_inpaint_negative_prompt,
    build_inpaint_prompt,
    build_prompt,
    build_reference_prompt,
    prompt_fits_tokenizers,
)


class WordTokenizer:
    model_max_length = CLIP_CONTEXT_TOKENS

    def __init__(self, extra_tokens=0):
        self.extra_tokens = extra_tokens

    def __call__(self, text, **kwargs):
        words = text.replace(",", " ").replace(";", " ").split()
        return {"input_ids": [0] * (len(words) + 2 + self.extra_tokens)}


def test_full_prompt_always_contains_architecture_protection():
    prompt = build_prompt()
    assert ARCHITECTURE_PREFIX in prompt
    assert "exact camera viewpoint" in prompt
    assert "architectural shell, room shape, walls and corners" in prompt
    assert "window and door position, count, size, shape and perspective" in prompt


def test_full_negative_removes_contradictory_layout_phrase():
    assert "identical layout" not in FULL_NEGATIVE_PROMPT.lower()
    assert "moved window" in FULL_NEGATIVE_PROMPT
    assert "changed wall geometry" in FULL_NEGATIVE_PROMPT


def test_all_supported_style_fragments_are_retained():
    assert set(STYLE_PROMPTS) == {
        "modern",
        "scandinavian",
        "industrial",
        "bohemian",
        "classic",
        "japandi",
        "luxury",
    }
    for key, fragment in STYLE_PROMPTS.items():
        assert fragment in build_prompt(style_key=key)


def test_user_text_extends_instead_of_replacing_protected_prompt():
    request = "warm indirect lighting and walnut finishes"
    prompt = build_prompt(style_key="modern", user_text=request)
    assert prompt.startswith(ARCHITECTURE_PREFIX)
    assert STYLE_PROMPTS["modern"] in prompt
    assert request in prompt
    assert prompt.index(STYLE_PROMPTS["modern"]) < prompt.index(request)


def test_unknown_style_does_not_remove_architecture_policy():
    prompt = build_prompt(style_key="unknown", user_text="a bright kitchen")
    assert ARCHITECTURE_PREFIX in prompt
    assert "a bright kitchen" in prompt


def test_full_prompt_fits_both_tokenizers_and_preserves_mandatory_content():
    tokenizers = (WordTokenizer(), WordTokenizer(extra_tokens=3))
    prompt = build_prompt(style_key="modern", tokenizers=tokenizers)
    assert prompt_fits_tokenizers(prompt, tokenizers)
    assert ARCHITECTURE_PREFIX in prompt
    assert STYLE_PROMPTS["modern"] in prompt
    assert "replace movable furniture and decor" in prompt


def test_long_user_request_is_shortened_without_truncating_protected_prompt():
    tokenizers = (WordTokenizer(), WordTokenizer(extra_tokens=4))
    request = " ".join(f"userword{index}" for index in range(100))
    prompt = build_prompt(style_key="modern", user_text=request, tokenizers=tokenizers)
    assert prompt_fits_tokenizers(prompt, tokenizers)
    assert ARCHITECTURE_PREFIX in prompt
    assert STYLE_PROMPTS["modern"] in prompt
    assert "replace movable furniture and decor" in prompt
    assert "userword0" in prompt
    assert "userword99" not in prompt


def test_full_negative_prompt_is_within_both_clip_contexts():
    tokenizers = (WordTokenizer(), WordTokenizer(extra_tokens=3))
    assert prompt_fits_tokenizers(FULL_NEGATIVE_PROMPT, tokenizers)


def test_reference_prompt_restores_final_notebook_fallback_semantics():
    prompt = build_reference_prompt()
    assert prompt.startswith(ARCHITECTURE_PREFIX)
    assert REFERENCE_REDESIGN_CORE in prompt
    assert REFERENCE_FALLBACK in prompt
    assert "full interior redesign" in prompt
    assert "interior design magazine photo" in prompt


def test_reference_user_text_extends_strong_prompt_and_stays_in_budget():
    tokenizers = (WordTokenizer(), WordTokenizer(extra_tokens=3))
    request = "warm oak palette " + " ".join(
        f"referenceword{index}" for index in range(100)
    )
    prompt = build_reference_prompt(user_text=request, tokenizers=tokenizers)
    assert prompt_fits_tokenizers(prompt, tokenizers)
    assert ARCHITECTURE_PREFIX in prompt
    assert REFERENCE_REDESIGN_CORE in prompt
    assert "warm oak palette" in prompt
    assert "referenceword99" not in prompt


def test_inpaint_prompt_is_mask_scoped_and_not_full_redesign():
    prompt = build_inpaint_prompt("bed", "new headboard and linen", "modern")
    assert "edit only the masked bed" in prompt
    assert "new headboard and linen" in prompt
    assert "preserve the walls, windows, doors" in prompt
    assert "full interior redesign" not in prompt.lower()


def test_inpaint_negative_is_target_aware():
    negative = build_inpaint_negative_prompt("sofa")
    assert "unchanged sofa" in negative
    assert "only recolored sofa" in negative
    assert "duplicate sofa" in negative
    assert "changed window" in negative
