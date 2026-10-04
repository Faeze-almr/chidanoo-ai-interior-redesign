"""Targeted inpainting edit endpoints.

Submits a targeted edit to the shared asynchronous worker.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.errors import CudaRequiredError, ParameterValidationError, StorageError
from app.core.final_config import TARGETED_INPAINT_CONFIG
from app.core.job_queue import job_queue
from app.core.history import save_history_entry
from app.core.model_manager import require_cuda
from app.core.pipeline import generate_region_result, validate_generation_parameters
from app.core.preprocessing import check_image_quality
from app.core.storage import save_generation_metadata, save_result_image, save_upload_image
from app.schemas.generate import GenerateRoomResponse

router = APIRouter(prefix="/edit", tags=["edit"])
logger = logging.getLogger(__name__)


@router.post("/region", response_model=GenerateRoomResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate_region(
    image: UploadFile,
    mask_image: UploadFile,
    style_key: Optional[str] = Form(default=None),
    user_text: Optional[str] = Form(default=None),
    depth_scale: float = Form(default=TARGETED_INPAINT_CONFIG.depth_scale),
    canny_scale: float = Form(default=TARGETED_INPAINT_CONFIG.canny_scale),
    num_inference_steps: int = Form(default=TARGETED_INPAINT_CONFIG.steps),
    guidance_scale: float = Form(default=TARGETED_INPAINT_CONFIG.guidance_scale),
    seed: int = Form(default=TARGETED_INPAINT_CONFIG.seed),
    target_description: str = Form(default="target area"),
    strength: float = Form(default=TARGETED_INPAINT_CONFIG.strength or 1.0),
) -> GenerateRoomResponse:
    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

    raw_mask = await mask_image.read()
    try:
        pil_mask = Image.open(io.BytesIO(raw_mask))
        pil_mask.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Uploaded mask file is not a readable image.")

    ok, message = check_image_quality(pil_image)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    try:
        validate_generation_parameters(
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            depth_scale=depth_scale,
            canny_scale=canny_scale,
            strength=strength,
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
        result = generate_region_result(
            pil_image,
            pil_mask,
            style_key=style_key,
            user_text=user_text,
            depth_scale=depth_scale,
            canny_scale=canny_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            target_description=target_description,
            strength=strength,
        )
        result_path = save_result_image(result.image, job.id)
        save_generation_metadata(result.metadata, job.id)
        try:
            save_history_entry(
                result,
                pil_image,
                style=style_key,
                user_request=user_text,
                mask_image=pil_mask,
            )
        except StorageError:
            logger.exception("Generation succeeded but history persistence failed")
        return str(result_path)

    job_queue.enqueue(job.id, run)

    return GenerateRoomResponse(job_id=job.id, status=job.status)
