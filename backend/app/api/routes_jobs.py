"""Job status polling and result retrieval/download.

Status is intentionally coarse (queued -> running -> succeeded/failed)
rather than a fabricated step-by-step percentage: that's what the job
queue can honestly observe without instrumenting the ported pipeline
functions with progress callbacks (see docs/architecture.md).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.job_queue import JobStatus, job_queue
from app.schemas.generate import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = job_queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        error=job.error,
        result_url=f"/api/jobs/{job.id}/image" if job.status == JobStatus.SUCCEEDED else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/{job_id}/image")
def get_job_image(job_id: str, download: bool = False) -> FileResponse:
    job = job_queue.get(job_id)
    if job is None or job.status != JobStatus.SUCCEEDED or job.result is None:
        raise HTTPException(status_code=404, detail="Result not available.")

    if download:
        return FileResponse(
            job.result,
            media_type="image/png",
            filename=f"chidanoo-{job_id}.png",
            content_disposition_type="attachment",
        )
    return FileResponse(job.result, media_type="image/png")
