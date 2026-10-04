"""GPU-resident model manager (singleton).

Owns the shared SDXL + dual-ControlNet pipeline, the inpaint pipeline built
from the same shared components, and the MiDaS depth detector.

Nothing is loaded until the corresponding getter (``get_pipe``,
``get_inpaint_pipe``, ``get_midas``) is called for the first time - merely
importing this module never downloads weights or touches the GPU.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import TYPE_CHECKING, Callable, Optional

import torch

from app.config import settings
from app.core.errors import CudaRequiredError, GenerationDomainError, ModelLoadError
from app.core.final_config import MODEL_CONFIG

if TYPE_CHECKING:
    from diffusers import (
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLControlNetPipeline,
    )

# CUDA allocator tuning used by the frozen runtime.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- Frozen model identifiers ---------------------------------------------

BASE_MODEL_ID = MODEL_CONFIG.base_model_id
DEPTH_CONTROLNET_ID = MODEL_CONFIG.depth_controlnet_id
CANNY_CONTROLNET_ID = MODEL_CONFIG.canny_controlnet_id
VAE_ID = MODEL_CONFIG.vae_id
MIDAS_REPO_ID = MODEL_CONFIG.midas_repo_id

logger = logging.getLogger(__name__)

# Full and Targeted share the canonical SDXL UNet. Reference mode releases
# those caches and builds a disposable pipeline, so all three modes still need
# one process-wide serialization guarantee for safe lifecycle transitions.
inference_lock = threading.RLock()


def get_device() -> str:
    """Returns the auto-detected execution device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: Optional[str] = None) -> torch.dtype:
    """Returns the frozen dtype for the selected device."""
    device = device or get_device()
    return torch.float16 if device == "cuda" else torch.float32


def require_cuda() -> None:
    """Rejects production generation before any model download/load on CPU."""

    if not torch.cuda.is_available():
        raise CudaRequiredError()


def _try_optimize(fn: Callable[[], None]) -> bool:
    """Enables an optional optimization without making it a requirement."""
    try:
        fn()
        return True
    except Exception:
        return False


def _load_component(model_class, model_id: str, prefer_fp16_variant: bool = True, **kwargs):
    """Loads a component with an fp16-variant fallback.

    Tries the fp16 ``variant`` first (half the fp32 size) and falls back to
    the regular checkpoint if that variant doesn't exist in the repo. Models
    are loaded without being moved to a device here - the caller decides
    when/whether to call ``.to(device)`` because model CPU offload manages
    device placement itself.
    """
    dtype = get_dtype()
    if prefer_fp16_variant:
        try:
            return model_class.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16",
                low_cpu_mem_usage=True,
                cache_dir=settings.model_cache_dir,
                **kwargs,
            )
        except Exception:
            pass  # fall back to the non-fp16-variant checkpoint below

    return model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        cache_dir=settings.model_cache_dir,
        **kwargs,
    )


class ModelManager:
    """Lazily loads and owns the shared model components.

    A single instance (``model_manager`` below) is used process-wide so the
    dual-ControlNet pipeline and the inpaint pipeline share the same
    underlying UNet/VAE/text-encoder weights so Full and Targeted avoid
    loading duplicate copies of the same canonical model components.
    """

    def __init__(self) -> None:
        self._pipe: Optional["StableDiffusionXLControlNetPipeline"] = None
        self._inpaint_pipe: Optional["StableDiffusionXLControlNetInpaintPipeline"] = None
        self._midas = None

    @property
    def device(self) -> str:
        return get_device()

    @property
    def dtype(self) -> torch.dtype:
        return get_dtype(self.device)

    @property
    def is_pipe_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def is_inpaint_pipe_loaded(self) -> bool:
        return self._inpaint_pipe is not None

    @property
    def is_midas_loaded(self) -> bool:
        return self._midas is not None

    def get_pipe(self) -> "StableDiffusionXLControlNetPipeline":
        """Returns the shared dual-ControlNet pipeline, loading it on first
        call. Includes both ControlNets, the fp16-fixed VAE, the base SDXL
        checkpoint, the Karras-sigma
        DPMSolver scheduler swap, the best-effort memory optimizations, and
        finally moving the whole pipeline to the active device."""
        if self._pipe is None:
            try:
                self._pipe = self._load_pipe()
            except GenerationDomainError:
                raise
            except Exception as exc:
                logger.exception("Failed to load the full redesign pipeline")
                raise ModelLoadError() from exc
        return self._pipe

    def get_inpaint_pipe(self) -> "StableDiffusionXLControlNetInpaintPipeline":
        """Returns the shared inpainting pipeline, built from the main
        pipe's own components (not separately loaded weights) on first
        call."""
        if self._inpaint_pipe is None:
            try:
                self._inpaint_pipe = self._build_inpaint_pipe(self.get_pipe())
            except GenerationDomainError:
                raise
            except Exception as exc:
                logger.exception("Failed to build the targeted inpaint pipeline")
                raise ModelLoadError() from exc
        return self._inpaint_pipe

    def get_midas(self):
        """Returns the shared MiDaS depth detector, loading it on first
        call. Loading stays here so preprocessing remains network/GPU-free
        on import."""
        if self._midas is None:
            try:
                self._midas = self._load_midas()
            except GenerationDomainError:
                raise
            except Exception as exc:
                logger.exception("Failed to load the depth estimator")
                raise ModelLoadError() from exc
        return self._midas

    def create_reference_pipe(self) -> "StableDiffusionXLControlNetPipeline":
        """Builds an uncached CPU-resident pipeline for one Reference run.

        Reference mode deliberately owns fresh SDXL/ControlNet/VAE objects so
        IP-Adapter attention processors can never mutate the canonical cached
        Full/Inpaint UNet. Device placement is deferred until after adapter
        loading, when the caller enables model CPU offload.
        """

        try:
            return self._build_reference_pipe()
        except GenerationDomainError:
            raise
        except Exception as exc:
            logger.exception("Failed to build the ephemeral reference pipeline")
            raise ModelLoadError("The reference-image pipeline could not be loaded.") from exc

    def unload_all(self) -> None:
        """Lifecycle hook: drops references to every owned component and
        frees GPU memory. Cleanup drops in-memory model references, runs garbage
        collection, and empties the CUDA cache. Everything reloads
        lazily on next access; nothing here is persisted across process
        restarts."""
        self._pipe = None
        self._inpaint_pipe = None
        self._midas = None
        gc.collect()
        if get_device() == "cuda":
            torch.cuda.empty_cache()

    # -- internal loaders ---------------------------------------------

    def _load_pipe(self) -> "StableDiffusionXLControlNetPipeline":
        require_cuda()
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionXLControlNetPipeline,
        )

        controlnet_depth = _load_component(ControlNetModel, DEPTH_CONTROLNET_ID)
        controlnet_canny = _load_component(ControlNetModel, CANNY_CONTROLNET_ID)
        vae = _load_component(AutoencoderKL, VAE_ID, prefer_fp16_variant=False)

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=[controlnet_depth, controlnet_canny],
            vae=vae,
            torch_dtype=self.dtype,
            variant="fp16",
            low_cpu_mem_usage=True,
            cache_dir=settings.model_cache_dir,
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=MODEL_CONFIG.use_karras_sigmas
        )

        _try_optimize(pipe.enable_attention_slicing)
        _try_optimize(pipe.vae.enable_tiling)
        _try_optimize(pipe.vae.enable_slicing)
        _try_optimize(pipe.enable_xformers_memory_efficient_attention)

        pipe = pipe.to(self.device)
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return pipe

    def _build_inpaint_pipe(
        self, pipe: "StableDiffusionXLControlNetPipeline"
    ) -> "StableDiffusionXLControlNetInpaintPipeline":
        from diffusers import StableDiffusionXLControlNetInpaintPipeline

        inpaint_pipe = StableDiffusionXLControlNetInpaintPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            unet=pipe.unet,
            controlnet=pipe.controlnet,
            scheduler=pipe.scheduler,
        )
        _try_optimize(inpaint_pipe.enable_attention_slicing)
        _try_optimize(inpaint_pipe.vae.enable_tiling)
        return inpaint_pipe

    def _build_reference_pipe(self) -> "StableDiffusionXLControlNetPipeline":
        require_cuda()
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            DPMSolverMultistepScheduler,
            StableDiffusionXLControlNetPipeline,
        )

        controlnet_depth = _load_component(ControlNetModel, DEPTH_CONTROLNET_ID)
        controlnet_canny = _load_component(ControlNetModel, CANNY_CONTROLNET_ID)
        vae = _load_component(AutoencoderKL, VAE_ID, prefer_fp16_variant=False)

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=[controlnet_depth, controlnet_canny],
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
            cache_dir=settings.model_cache_dir,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=MODEL_CONFIG.use_karras_sigmas
        )

        # Keep default attention processors intact until IP-Adapter has loaded.
        # VAE tiling/slicing do not replace UNet attention processors.
        _try_optimize(pipe.vae.enable_tiling)
        _try_optimize(pipe.vae.enable_slicing)
        return pipe

    def _load_midas(self):
        require_cuda()
        from controlnet_aux import MidasDetector

        midas = MidasDetector.from_pretrained(MIDAS_REPO_ID, cache_dir=settings.model_cache_dir)
        return midas.to(self.device)


model_manager = ModelManager()
