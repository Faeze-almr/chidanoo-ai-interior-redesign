"""Optional Gradio entrypoint for the canonical CHIDANOO workflows."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.gradio_ui import build_demo  # noqa: E402


def main(*, share: bool = False) -> None:
    demo = build_demo()
    demo.queue(max_size=20).launch(share=share)


if __name__ == "__main__":
    main()
