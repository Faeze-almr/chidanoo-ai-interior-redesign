"""Thin Gradio UI for the canonical CHIDANOO core.

Gradio is imported only by :func:`build_demo`, so the FastAPI backend and its
tests do not acquire an import-time UI dependency. All inference is delegated
to the canonical result APIs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageChops

from app.core.errors import GenerationDomainError, StorageError
from app.core.final_config import (
    FULL_REDESIGN_CONFIG,
    REFERENCE_FEATURE_CONFIG,
    REFERENCE_REDESIGN_CONFIG,
    TARGETED_INPAINT_CONFIG,
)
from app.core.history import (
    HISTORY_DIR,
    export_history_zip,
    history_views,
    load_history_output,
    save_history_entry,
)
from app.core.pipeline import (
    GenerationResult,
    generate_reference_result,
    generate_region_result,
    generate_room_result,
)

logger = logging.getLogger(__name__)

STYLE_LABELS = {
    "Modern": "modern",
    "Scandinavian": "scandinavian",
    "Industrial": "industrial",
    "Bohemian": "bohemian",
    "Classic": "classic",
    "Japandi": "japandi",
    "Luxury": "luxury",
}
NO_STYLE_LABEL = "Prompt only"
STYLE_CHOICES = [NO_STYLE_LABEL, *STYLE_LABELS]


def _style_key(label: str | None) -> str | None:
    return STYLE_LABELS.get(label or "")


def _seed(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return FULL_REDESIGN_CONFIG.seed
    return parsed if parsed >= 0 else FULL_REDESIGN_CONFIG.seed


def _pil(value) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    try:
        return Image.fromarray(value)
    except Exception:
        return None


def extract_mask_from_editor(value) -> Image.Image | None:
    """Combines every painted ImageEditor layer into a binary alpha mask."""

    if not isinstance(value, dict) or _pil(value.get("background")) is None:
        return None
    layers = value.get("layers") or []
    combined = None
    for layer_value in layers:
        layer = _pil(layer_value)
        if layer is None:
            continue
        alpha = layer.convert("RGBA").getchannel("A")
        combined = alpha if combined is None else ImageChops.lighter(combined, alpha)
    if combined is None:
        return None
    mask = combined.point(lambda pixel: 255 if pixel >= 128 else 0, mode="L")
    return mask if mask.getbbox() is not None else None


def _history_output_path(record: dict) -> str:
    return str(HISTORY_DIR / record["id"] / record["output_path"])


def _save_result(
    result: GenerationResult,
    input_image: Image.Image,
    *,
    style: str | None,
    user_request: str | None,
    reference_image: Image.Image | None = None,
    mask_image: Image.Image | None = None,
) -> tuple[str | None, str | None]:
    try:
        record = save_history_entry(
            result,
            input_image,
            style=style,
            user_request=user_request,
            reference_image=reference_image,
            mask_image=mask_image,
        )
        return _history_output_path(record), None
    except StorageError as exc:
        logger.exception("Generation succeeded but history persistence failed")
        return None, exc.user_message


def _safe_failure(exc: BaseException) -> str:
    if isinstance(exc, GenerationDomainError):
        return exc.user_message
    logger.exception("Unexpected Gradio callback failure")
    return "The operation failed unexpectedly. Please try again."


def run_full(image, style_label, text, depth, canny, steps, guidance, seed):
    if _pil(image) is None:
        return None, "Please provide an interior image.", None, "", None
    source = _pil(image).convert("RGB")
    try:
        result = generate_room_result(
            source,
            style_key=_style_key(style_label),
            user_text=text,
            depth_scale=float(depth),
            canny_scale=float(canny),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            seed=_seed(seed),
        )
        output_path, history_error = _save_result(
            result,
            source,
            style=_style_key(style_label),
            user_request=text,
        )
        status = f"Done in {result.metadata.runtime_sec:.1f}s."
        if history_error:
            status += f" History warning: {history_error}"
        else:
            status += " Saved to local history."
        return (source, result.image), status, result.image, result.metadata.prompt, output_path
    except Exception as exc:
        return None, _safe_failure(exc), None, "", None


def run_edit(editor, target, style_label, text, depth, canny, steps, guidance, seed):
    if not isinstance(editor, dict) or _pil(editor.get("background")) is None:
        return None, "Please provide an interior image.", None, "", None
    source = _pil(editor["background"]).convert("RGB")
    mask = extract_mask_from_editor(editor)
    if mask is None:
        return None, "Paint the area that should be edited.", None, "", None
    try:
        result = generate_region_result(
            source,
            mask,
            style_key=_style_key(style_label),
            user_text=text,
            target_description=(target or "target area").strip() or "target area",
            depth_scale=float(depth),
            canny_scale=float(canny),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            seed=_seed(seed),
            strength=TARGETED_INPAINT_CONFIG.strength or 1.0,
        )
        output_path, history_error = _save_result(
            result,
            source,
            style=_style_key(style_label),
            user_request=text,
            mask_image=mask,
        )
        status = f"Targeted edit completed in {result.metadata.runtime_sec:.1f}s."
        if history_error:
            status += f" History warning: {history_error}"
        else:
            status += " Saved to local history."
        return (source, result.image), status, result.image, result.metadata.prompt, output_path
    except Exception as exc:
        return None, _safe_failure(exc), None, "", None


def run_reference(room, reference, style_label, text, scale, depth, canny, steps, guidance, seed):
    source = _pil(room)
    reference_image = _pil(reference)
    if source is None or reference_image is None:
        return None, "Both an interior image and a reference image are required.", None, "", None
    source = source.convert("RGB")
    reference_image = reference_image.convert("RGB")
    try:
        result = generate_reference_result(
            source,
            reference_image,
            style_key=_style_key(style_label),
            user_text=text,
            reference_scale=float(scale),
            depth_scale=float(depth),
            canny_scale=float(canny),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            seed=_seed(seed),
        )
        output_path, history_error = _save_result(
            result,
            source,
            style=_style_key(style_label),
            user_request=text,
            reference_image=reference_image,
        )
        status = f"Reference redesign completed in {result.metadata.runtime_sec:.1f}s."
        if history_error:
            status += f" History warning: {history_error}"
        else:
            status += " Saved to local history."
        return (source, result.image), status, result.image, result.metadata.prompt, output_path
    except Exception as exc:
        return None, _safe_failure(exc), None, "", None


def refresh_history_data():
    try:
        return history_views()
    except StorageError as exc:
        logger.exception("History refresh failed")
        return [], [], []


def load_selected_history(entry_id):
    if not entry_id:
        return None, "Select a history entry first."
    try:
        return load_history_output(entry_id), f"Loaded {entry_id}."
    except StorageError as exc:
        logger.exception("History output load failed")
        return None, exc.user_message


def export_history():
    try:
        return str(export_history_zip())
    except StorageError:
        logger.exception("History ZIP export failed")
        return None


def build_demo():
    """Builds the Gradio workflow around canonical backend APIs."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            'Gradio UI dependencies are not installed. Install the backend with the "ui" extra.'
        ) from exc

    css_path = Path(__file__).resolve().parents[2] / "assets" / "gradio_styles.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    with gr.Blocks(css=css, title="CHIDANOO — Interior AI") as demo:
        gr.HTML(
            "<div class='hero'><h1>CHIDANOO — Interior AI</h1>"
            "<p>Structure-aware redesign • targeted editing • reference style • local history</p></div>"
        )
        with gr.Tabs():
            with gr.Tab("Full Redesign"):
                with gr.Row():
                    with gr.Column():
                        full_input = gr.Image(type="pil", label="Interior image")
                        full_style = gr.Dropdown(STYLE_CHOICES, value="Modern", label="Style")
                        full_text = gr.Textbox(label="Optional request", lines=2)
                        with gr.Accordion("Advanced settings", open=False):
                            full_depth = gr.Slider(0, 1, FULL_REDESIGN_CONFIG.depth_scale, 0.05, label="Depth scale")
                            full_canny = gr.Slider(0, 1, FULL_REDESIGN_CONFIG.canny_scale, 0.05, label="Canny scale")
                            full_steps = gr.Slider(1, 50, FULL_REDESIGN_CONFIG.steps, 1, label="Steps")
                            full_cfg = gr.Slider(0, 12, FULL_REDESIGN_CONFIG.guidance_scale, 0.5, label="CFG")
                            full_seed = gr.Number(FULL_REDESIGN_CONFIG.seed, precision=0, label="Seed")
                        full_button = gr.Button("Redesign", variant="primary")
                    with gr.Column():
                        full_compare = gr.ImageSlider(label="Before / after", type="pil")
                        full_status = gr.Textbox(label="Status", interactive=False)
                        full_output = gr.Image(type="pil", label="Output")
                        full_prompt = gr.Textbox(label="Final prompt", interactive=False)
                        full_download = gr.File(label="Download")
                full_button.click(
                    run_full,
                    [full_input, full_style, full_text, full_depth, full_canny, full_steps, full_cfg, full_seed],
                    [full_compare, full_status, full_output, full_prompt, full_download],
                )

            with gr.Tab("Targeted Inpainting"):
                with gr.Row():
                    with gr.Column():
                        edit_input = gr.ImageEditor(type="pil", label="Interior image and mask")
                        edit_target = gr.Textbox(value="furniture", label="Masked target")
                        edit_style = gr.Dropdown(STYLE_CHOICES, value=NO_STYLE_LABEL, label="Style")
                        edit_text = gr.Textbox(label="Requested change", lines=2)
                        edit_depth = gr.Slider(0, 1, TARGETED_INPAINT_CONFIG.depth_scale, 0.05, label="Depth scale")
                        edit_canny = gr.Slider(0, 1, TARGETED_INPAINT_CONFIG.canny_scale, 0.05, label="Canny scale")
                        edit_steps = gr.Slider(1, 50, TARGETED_INPAINT_CONFIG.steps, 1, label="Steps")
                        edit_cfg = gr.Slider(0, 12, TARGETED_INPAINT_CONFIG.guidance_scale, 0.5, label="CFG")
                        edit_seed = gr.Number(TARGETED_INPAINT_CONFIG.seed, precision=0, label="Seed")
                        edit_button = gr.Button("Edit masked target", variant="primary")
                    with gr.Column():
                        edit_compare = gr.ImageSlider(label="Before / after", type="pil")
                        edit_status = gr.Textbox(label="Status", interactive=False)
                        edit_output = gr.Image(type="pil", label="Protected output")
                        edit_prompt = gr.Textbox(label="Final prompt", interactive=False)
                        edit_download = gr.File(label="Download")
                edit_button.click(
                    run_edit,
                    [edit_input, edit_target, edit_style, edit_text, edit_depth, edit_canny, edit_steps, edit_cfg, edit_seed],
                    [edit_compare, edit_status, edit_output, edit_prompt, edit_download],
                )

            with gr.Tab("Reference Image"):
                with gr.Row():
                    with gr.Column():
                        ref_room = gr.Image(type="pil", label="Interior image")
                        ref_image = gr.Image(type="pil", label="Style reference")
                        ref_style = gr.Dropdown(STYLE_CHOICES, value=NO_STYLE_LABEL, label="Optional style")
                        ref_text = gr.Textbox(label="Optional request", lines=2)
                        ref_scale = gr.Slider(0, 1, REFERENCE_FEATURE_CONFIG.default_scale, 0.05, label="Reference scale")
                        ref_depth = gr.Slider(0, 1, REFERENCE_REDESIGN_CONFIG.depth_scale, 0.05, label="Depth scale")
                        ref_canny = gr.Slider(0, 1, REFERENCE_REDESIGN_CONFIG.canny_scale, 0.05, label="Canny scale")
                        ref_steps = gr.Slider(1, 50, REFERENCE_REDESIGN_CONFIG.steps, 1, label="Steps")
                        ref_cfg = gr.Slider(0, 12, REFERENCE_REDESIGN_CONFIG.guidance_scale, 0.5, label="CFG")
                        ref_seed = gr.Number(REFERENCE_REDESIGN_CONFIG.seed, precision=0, label="Seed")
                        ref_button = gr.Button("Redesign with reference", variant="primary")
                    with gr.Column():
                        ref_compare = gr.ImageSlider(label="Before / after", type="pil")
                        ref_status = gr.Textbox(label="Status", interactive=False)
                        ref_output = gr.Image(type="pil", label="Output")
                        ref_prompt = gr.Textbox(label="Final prompt", interactive=False)
                        ref_download = gr.File(label="Download")
                ref_button.click(
                    run_reference,
                    [ref_room, ref_image, ref_style, ref_text, ref_scale, ref_depth, ref_canny, ref_steps, ref_cfg, ref_seed],
                    [ref_compare, ref_status, ref_output, ref_prompt, ref_download],
                )

            with gr.Tab("History"):
                refresh_button = gr.Button("Refresh history")
                gallery = gr.Gallery(label="Gallery", columns=3, height=500)
                table = gr.Dataframe(label="Metadata", interactive=False)
                history_choice = gr.Dropdown(label="Previous output")
                load_button = gr.Button("Load into Full Redesign")
                loaded_status = gr.Textbox(label="Status", interactive=False)
                export_button = gr.Button("Export history ZIP")
                export_file = gr.File(label="History ZIP")

                def refresh_for_gradio():
                    gallery_data, rows, choices = refresh_history_data()
                    selected = choices[0] if choices else None
                    return gallery_data, rows, gr.Dropdown(choices=choices, value=selected)

                refresh_button.click(refresh_for_gradio, outputs=[gallery, table, history_choice])
                load_button.click(
                    load_selected_history,
                    inputs=[history_choice],
                    outputs=[full_input, loaded_status],
                )
                export_button.click(export_history, outputs=export_file)
                demo.load(refresh_for_gradio, outputs=[gallery, table, history_choice])

            with gr.Tab("Help"):
                gr.Markdown(
                    """## Workflow notes

- Full Redesign always retains the frozen architecture-protection policy.
- Targeted Inpainting edits white mask pixels and restores every protected pixel exactly.
- Reference Image loads IP-Adapter only for that operation and restores the shared SDXL state afterward.
- History is local file storage; export the ZIP before ending an ephemeral Kaggle/Colab session.
"""
                )
    return demo
