from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.core.job_queue import JobStatus


class GenerateRoomResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    error: Optional[str] = None
    result_url: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
