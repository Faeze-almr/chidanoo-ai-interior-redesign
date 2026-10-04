"""Isolated IP-Adapter lifecycle for ephemeral Reference pipelines."""

from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Iterator

import torch

from app.config import settings
from app.core.errors import (
    FeatureCleanupError,
    FeatureLoadError,
    GenerationOutOfMemoryError,
    ModelLoadError,
)
from app.core.final_config import REFERENCE_FEATURE_CONFIG

logger = logging.getLogger(__name__)

REFERENCE_CLEANUP_MAX_BYTES = 1024**3


class AdapterLoadState(str, Enum):
    NOT_STARTED = "not_started"
    LOADING = "loading"
    LOADED = "loaded"


class ReferenceAdapterManager:
    """Owns adapter state for one disposable Reference pipeline at a time."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_pipe = None
        self._state = AdapterLoadState.NOT_STARTED
        self._last_cleanup_allocated = 0
        self._last_cleanup_reserved = 0
        self._cleanup_allows_full_reload = True

    @property
    def state(self) -> AdapterLoadState:
        return self._state

    @property
    def is_loaded(self) -> bool:
        return self._state is AdapterLoadState.LOADED

    @property
    def last_cleanup_allocated(self) -> int:
        return self._last_cleanup_allocated

    @property
    def last_cleanup_reserved(self) -> int:
        return self._last_cleanup_reserved

    @property
    def cleanup_allows_full_reload(self) -> bool:
        return self._cleanup_allows_full_reload

    @staticmethod
    def _drop_feature_references(pipe) -> None:
        if pipe is None:
            return
        for attribute in ("image_encoder", "feature_extractor"):
            if hasattr(pipe, attribute):
                try:
                    setattr(pipe, attribute, None)
                except Exception:
                    logger.exception("Failed to clear Reference feature attribute %s", attribute)
        unet = getattr(pipe, "unet", None)
        if unet is not None and hasattr(unet, "encoder_hid_proj"):
            try:
                unet.encoder_hid_proj = None
                if getattr(unet, "config", None) is not None:
                    unet.config.encoder_hid_dim_type = None
            except Exception:
                logger.exception("Failed to clear Reference encoder projection")

    @staticmethod
    def _clear_cuda_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.exception("CUDA cache release failed during Reference cleanup")
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                try:
                    ipc_collect()
                except Exception:
                    logger.exception("CUDA IPC cleanup failed during Reference cleanup")

    @staticmethod
    def _adapter_projection_input_dim(pipe) -> int | None:
        encoder_projection = getattr(getattr(pipe, "unet", None), "encoder_hid_proj", None)
        projection_layers = getattr(encoder_projection, "image_projection_layers", None)
        projection = projection_layers[0] if projection_layers else encoder_projection
        direct_projection = getattr(projection, "image_embeds", None)
        direct_input = getattr(direct_projection, "in_features", None)
        if direct_input is not None:
            return int(direct_input)

        modules = getattr(projection, "modules", None)
        if callable(modules):
            for module in modules():
                input_features = getattr(module, "in_features", None)
                if input_features is not None:
                    return int(input_features)
        return None

    @classmethod
    def _validate_encoder_adapter_compatibility(cls, pipe) -> None:
        image_encoder = getattr(pipe, "image_encoder", None)
        encoder_config = getattr(image_encoder, "config", None)
        hidden_size = getattr(encoder_config, "hidden_size", None)
        projection_dim = getattr(encoder_config, "projection_dim", None)
        if projection_dim is None:
            projection_dim = getattr(getattr(image_encoder, "visual_projection", None), "out_features", None)
        adapter_input_dim = cls._adapter_projection_input_dim(pipe)
        encoder_class = type(image_encoder).__name__ if image_encoder is not None else None
        # Diffusers resolves image_encoder_folder from the adapter repository
        # root. It is independent of the adapter-weight subfolder.
        resolved_folder = (
            f"{REFERENCE_FEATURE_CONFIG.adapter_id}/"
            f"{REFERENCE_FEATURE_CONFIG.image_encoder_folder}"
        )
        logger.info(
            "Reference encoder compatibility: path=%s class=%s hidden_size=%s "
            "projection_dim=%s adapter_input_dim=%s",
            resolved_folder,
            encoder_class,
            hidden_size,
            projection_dim,
            adapter_input_dim,
        )

        if image_encoder is None or projection_dim is None:
            raise FeatureLoadError(
                "The Reference image encoder dimensions could not be verified."
            )
        projection_dim = int(projection_dim)
        expected_projection_dim = REFERENCE_FEATURE_CONFIG.expected_projection_dim
        if projection_dim != expected_projection_dim:
            raise FeatureLoadError(
                "The Reference image encoder is incompatible with "
                f"{REFERENCE_FEATURE_CONFIG.weight_name} "
                f"(projection_dim={projection_dim}, expected "
                f"{expected_projection_dim})."
            )
        if adapter_input_dim is not None and projection_dim != adapter_input_dim:
            raise FeatureLoadError(
                "The Reference image encoder and IP-Adapter projection are incompatible "
                f"({projection_dim} != {adapter_input_dim})."
            )

    @classmethod
    def _remove_pipeline_hooks(cls, pipe) -> None:
        if pipe is None:
            return
        maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
        if callable(maybe_free):
            try:
                maybe_free()
            except Exception:
                logger.exception("Diffusers model-hook release failed")

        remove_all_hooks = getattr(pipe, "remove_all_hooks", None)
        if callable(remove_all_hooks):
            try:
                remove_all_hooks()
            except Exception:
                logger.exception("Diffusers hook removal failed")

        try:
            from accelerate.hooks import remove_hook_from_module

            components = getattr(pipe, "components", {})
            seen: set[int] = set()
            for component in components.values():
                if component is None or id(component) in seen or not hasattr(component, "modules"):
                    continue
                seen.add(id(component))
                try:
                    remove_hook_from_module(component, recurse=True)
                except TypeError:
                    remove_hook_from_module(component)
                except Exception:
                    logger.exception("Accelerate hook removal failed for %s", type(component).__name__)
        except Exception:
            logger.exception("Accelerate hook cleanup was unavailable")

    def release_pipeline(self, pipe) -> None:
        """Best-effort hook and strong-reference cleanup for an ephemeral pipe."""

        with self._lock:
            self._remove_pipeline_hooks(pipe)
            self._drop_feature_references(pipe)
            self._reset_state()
            self._clear_cuda_cache()

    def record_post_cleanup_memory(self) -> None:
        """Records whether a clean canonical reload is safe after Reference."""

        allocated = 0
        reserved = 0
        healthy = True
        if torch.cuda.is_available():
            try:
                allocated = int(torch.cuda.memory_allocated())
                reserved = int(torch.cuda.memory_reserved())
                healthy = max(allocated, reserved) <= REFERENCE_CLEANUP_MAX_BYTES
            except Exception:
                logger.exception("Reference cleanup CUDA memory inspection failed")
                healthy = False
        self._last_cleanup_allocated = allocated
        self._last_cleanup_reserved = reserved
        self._cleanup_allows_full_reload = healthy
        logger.info(
            "Reference cleanup CUDA memory: allocated_bytes=%d reserved_bytes=%d "
            "full_reload_safe=%s",
            allocated,
            reserved,
            healthy,
        )

    @staticmethod
    def _disable_slicing_and_verify(pipe) -> None:
        disable_slicing = getattr(pipe, "disable_attention_slicing", None)
        if callable(disable_slicing):
            try:
                disable_slicing()
            except Exception as exc:
                raise ModelLoadError(
                    "Reference attention could not be prepared for the image adapter."
                ) from exc

        processors = getattr(getattr(pipe, "unet", None), "attn_processors", {})
        sliced = [
            name
            for name, processor in dict(processors).items()
            if type(processor).__name__.startswith("SlicedAttnProcessor")
        ]
        if sliced:
            raise ModelLoadError(
                "Reference attention slicing remained active before image-adapter loading."
            )

    def _reset_state(self) -> None:
        self._active_pipe = None
        self._state = AdapterLoadState.NOT_STARTED

    def _discard_partial_load(self) -> None:
        """Drops a partially mutated disposable pipe without unsafe unload."""

        pipe = self._active_pipe
        self._drop_feature_references(pipe)
        self._reset_state()
        self._clear_cuda_cache()

    def _cleanup_loaded(self) -> None:
        pipe = self._active_pipe
        cleanup_error: BaseException | None = None
        try:
            if pipe is not None and self._state is AdapterLoadState.LOADED:
                unload = getattr(pipe, "unload_ip_adapter", None)
                if callable(unload):
                    try:
                        unload()
                    except Exception as exc:
                        cleanup_error = exc
                        logger.exception("Best-effort IP-Adapter unload failed")
        finally:
            self._drop_feature_references(pipe)
            self._reset_state()
            self._clear_cuda_cache()

        if cleanup_error is not None:
            raise FeatureCleanupError() from cleanup_error

    def load_feature(self, pipe, scale: float = REFERENCE_FEATURE_CONFIG.default_scale):
        """Loads IP-Adapter only into an uncached, CPU-built Reference pipe."""

        with self._lock:
            return self._load_feature_unlocked(pipe, scale)

    def _load_feature_unlocked(
        self,
        pipe,
        scale: float = REFERENCE_FEATURE_CONFIG.default_scale,
    ):
        if self._state is not AdapterLoadState.NOT_STARTED:
            raise ModelLoadError("Reference-image mode is already active.")

        self._last_cleanup_allocated = 0
        self._last_cleanup_reserved = 0
        self._cleanup_allows_full_reload = True

        # Diffusers' IP-Adapter conversion must see default attention
        # processors, never SlicedAttnProcessor instances.
        self._disable_slicing_and_verify(pipe)
        self._active_pipe = pipe
        self._state = AdapterLoadState.LOADING

        try:
            pipe.load_ip_adapter(
                REFERENCE_FEATURE_CONFIG.adapter_id,
                subfolder=REFERENCE_FEATURE_CONFIG.subfolder,
                weight_name=REFERENCE_FEATURE_CONFIG.weight_name,
                image_encoder_folder=REFERENCE_FEATURE_CONFIG.image_encoder_folder,
                cache_dir=settings.model_cache_dir,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            logger.exception("Failed to load the reference-image IP-Adapter")
            # load_ip_adapter can fail after mutating attention processors.
            # The pipe is disposable, so never invoke unload on this state.
            self._discard_partial_load()
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise GenerationOutOfMemoryError() from exc
            raise ModelLoadError("The reference-image adapter could not be loaded.") from exc

        self._state = AdapterLoadState.LOADED
        try:
            self._validate_encoder_adapter_compatibility(pipe)
            pipe.set_ip_adapter_scale(float(scale))
            enable_offload = getattr(pipe, "enable_model_cpu_offload", None)
            if not callable(enable_offload):
                raise RuntimeError("model CPU offload is unavailable")
            enable_offload()
        except Exception as exc:
            logger.exception("Failed to configure the loaded Reference pipeline")
            try:
                self._cleanup_loaded()
            except FeatureCleanupError:
                logger.exception("Reference cleanup also failed after configuration failure")
            if isinstance(exc, FeatureLoadError):
                raise
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise GenerationOutOfMemoryError() from exc
            raise ModelLoadError("The reference-image pipeline could not be configured.") from exc
        return pipe

    def unload_feature(self) -> None:
        with self._lock:
            if self._state is AdapterLoadState.LOADING:
                self._discard_partial_load()
            else:
                self._cleanup_loaded()

    @contextmanager
    def feature_session(
        self,
        pipe,
        scale: float = REFERENCE_FEATURE_CONFIG.default_scale,
    ) -> Iterator[object]:
        """Runs one serialized operation and discards all adapter references."""

        with self._lock:
            active_pipe = self._load_feature_unlocked(pipe, scale)
            try:
                yield active_pipe
            finally:
                self._cleanup_loaded()


reference_adapter_manager = ReferenceAdapterManager()
