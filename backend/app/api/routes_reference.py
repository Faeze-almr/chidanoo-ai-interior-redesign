"""Reference-image (IP-Adapter) redesign endpoints.

Submits reference-guided redesign to the shared asynchronous worker.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.errors import CudaRequiredError, ParameterValidationError, StorageError
from app.core.final_config import REFERENCE_REDESIGN_CONFIG, REFERENCE_FEATURE_CONFIG
from app.core.job_queue import job_queue
from app.core.history import save_history_entry
from app.core.model_manager import require_cuda
from app.core.pipeline import generate_reference_result, validate_generation_parameters
from app.core.preprocessing import check_image_quality
from app.core.storage import save_generation_metadata, save_result_image, save_upload_image
from app.schemas.generate import GenerateRoomResponse

router = APIRouter(prefix="/reference", tags=["reference"])
logger = logging.getLogger(__name__)


@router.post("/room", response_model=GenerateRoomResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_reference(
    image: UploadFile,
    reference_image: UploadFile,
    style_key: Optional[str] = Form(default=None),
    user_text: Optional[str] = Form(default=None),
    depth_scale: float = Form(default=REFERENCE_REDESIGN_CONFIG.depth_scale),
    canny_scale: float = Form(default=REFERENCE_REDESIGN_CONFIG.canny_scale),
    reference_scale: float = Form(default=REFERENCE_FEATURE_CONFIG.default_scale),
    num_inference_steps: int = Form(default=REFERENCE_REDESIGN_CONFIG.steps),
    guidance_scale: float = Form(default=REFERENCE_REDESIGN_CONFIG.guidance_scale),
    seed: int = Form(default=REFERENCE_REDESIGN_CONFIG.seed),
) -> GenerateRoomResponse:
    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

    raw_ref = await reference_image.read()
    try:
        pil_ref = Image.open(io.BytesIO(raw_ref))
        pil_ref.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Uploaded reference file is not a readable image.")

    ok, message = check_image_quality(pil_image)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    reference_ok, reference_message = check_image_quality(pil_ref)
    if not reference_ok:
        raise HTTPException(status_code=400, detail=f"Reference image: {reference_message}")

    try:
        validate_generation_parameters(
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            depth_scale=depth_scale,
            canny_scale=canny_scale,
            reference_scale=reference_scale,
        )
    except ParameterValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from exc

    try:
        require_cuda()
    except CudaRequiredError as exc:
        raise HTTPException(status_code=503, detail=exc.user_message) from exc

    job = job_queue.create()
    try:
        save_upload_image(pil_image, job.id)
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=exc.user_message) from exc

    def run() -> str:
        result = generate_reference_result(
            pil_image,
            pil_ref,
            style_key=style_key,
            user_text=user_text,
            depth_scale=depth_scale,
            canny_scale=canny_scale,
            reference_scale=reference_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        result_path = save_result_image(result.image, job.id)
        save_generation_metadata(result.metadata, job.id)
        try:
            save_history_entry(
                result,
                pil_image,
                style=style_key,
                user_request=user_text,
                reference_image=pil_ref,
            )
        except StorageError:
            logger.exception("Generation succeeded but history persistence failed")
        return str(result_path)

    job_queue.enqueue(job.id, run)

    return GenerateRoomResponse(job_id=job.id, status=job.status)
