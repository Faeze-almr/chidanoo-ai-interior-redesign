import numpy as np
import sys
from types import SimpleNamespace

from PIL import Image

from app import gradio_ui


def test_gradio_module_imports_without_gradio_installed():
    assert callable(gradio_ui.build_demo)
    assert "Japandi" in gradio_ui.STYLE_CHOICES
    assert "Luxury" in gradio_ui.STYLE_CHOICES


def test_editor_mask_combines_layers_and_binarizes():
    background = Image.new("RGB", (20, 20), "gray")
    first = np.zeros((20, 20, 4), dtype=np.uint8)
    second = np.zeros((20, 20, 4), dtype=np.uint8)
    first[2:6, 2:6, 3] = 255
    second[10:15, 10:15, 3] = 200
    mask = gradio_ui.extract_mask_from_editor(
        {
            "background": background,
            "layers": [Image.fromarray(first), Image.fromarray(second)],
        }
    )
    assert mask is not None
    assert set(np.unique(np.asarray(mask))) <= {0, 255}
    assert mask.getpixel((3, 3)) == 255
    assert mask.getpixel((12, 12)) == 255
    assert mask.getpixel((0, 0)) == 0


def test_gradio_callback_returns_safe_message_for_unexpected_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("secret stack detail")

    monkeypatch.setattr(gradio_ui, "generate_room_result", fail)
    image = Image.new("RGB", (512, 512), "gray")
    _, status, output, prompt, download = gradio_ui.run_full(
        image, "Modern", "request", 0.5, 0.2, 30, 7.5, 42
    )
    assert status == "The operation failed unexpectedly. Please try again."
    assert "secret" not in status
    assert output is None
    assert prompt == ""
    assert download is None


def test_all_five_gradio_tabs_still_build(monkeypatch):
    tab_labels = []

    class Component:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def click(self, *args, **kwargs):
            return self

        def load(self, *args, **kwargs):
            return self

    class Tab(Component):
        def __init__(self, label, *args, **kwargs):
            tab_labels.append(label)

    fake_gradio = SimpleNamespace(
        Blocks=Component,
        HTML=Component,
        Tabs=Component,
        Tab=Tab,
        Row=Component,
        Column=Component,
        Image=Component,
        Dropdown=Component,
        Textbox=Component,
        Accordion=Component,
        Slider=Component,
        Number=Component,
        Button=Component,
        ImageSlider=Component,
        File=Component,
        ImageEditor=Component,
        Gallery=Component,
        Dataframe=Component,
        Markdown=Component,
    )
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)

    demo = gradio_ui.build_demo()

    assert demo is not None
    assert tab_labels == [
        "Full Redesign",
        "Targeted Inpainting",
        "Reference Image",
        "History",
        "Help",
    ]
