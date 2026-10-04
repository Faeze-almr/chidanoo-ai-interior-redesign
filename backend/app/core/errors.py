"""Domain-level exceptions with safe user-facing messages."""

from __future__ import annotations


class GenerationDomainError(RuntimeError):
    """Base error whose string form is safe to return to a client."""

    default_message = "Generation could not be completed."

    def __init__(self, message: str | None = None) -> None:
        self.user_message = message or self.default_message
        super().__init__(self.user_message)


class InputValidationError(GenerationDomainError):
    default_message = "The input image is invalid."


class MaskValidationError(GenerationDomainError):
    default_message = "The edit mask is invalid."


class ParameterValidationError(GenerationDomainError):
    default_message = "One or more generation settings are invalid."


class CudaRequiredError(GenerationDomainError):
    default_message = "GPU/CUDA is required for this generation pipeline."


class ModelLoadError(GenerationDomainError):
    default_message = "The generation model could not be loaded."


class FeatureLoadError(ModelLoadError):
    default_message = "The reference-image feature could not be loaded safely."


class GenerationOutOfMemoryError(GenerationDomainError):
    default_message = "The GPU ran out of memory while generating the image."


class GenerationRuntimeError(GenerationDomainError):
    default_message = "Generation failed while processing the image."


class FeatureCleanupError(GenerationRuntimeError):
    default_message = "The optional feature could not restore the generation model safely."


class StorageError(GenerationDomainError):
    default_message = "The generated file could not be saved."
