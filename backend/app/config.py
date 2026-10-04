"""Application settings and project-relative paths.

All filesystem paths here are derived from this file's own location via
``pathlib`` - never from the process's current working directory, and never
hard-coded to a workstation- or notebook-specific absolute path.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent

STORAGE_DIR = BACKEND_DIR / "storage"
OUTPUTS_DIR = REPO_ROOT / "outputs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CHIDANOO"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000

    # Reserved/informational: production generation auto-detects CUDA.
    device: str = "cuda"

    # Optional override for where Hugging Face model downloads are cached.
    # None uses huggingface_hub's default cache location.
    model_cache_dir: Optional[str] = None

    cors_origins: list[str] = ["http://localhost:5173"]

settings = Settings()

# Ensure local data directories exist on a fresh checkout.
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
