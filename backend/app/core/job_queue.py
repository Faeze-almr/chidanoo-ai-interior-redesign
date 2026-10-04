"""In-process job queue for long-running generation requests.

A single background worker consumes jobs from an ``asyncio.Queue`` one at a
time. Because exactly one job is ever being processed, this alone
guarantees only one generation call ever touches the GPU at once - the
"single GPU lock" the approved plan calls for, without a separate lock
object, and without Redis/Celery/Kafka or any other broker. This only needs
to survive the lifetime of one running process, which matches this
project's single-user, single-GPU academic scope.

Job state lives in memory only. Completed outputs use the separate file-backed
history service. This module supports progress polling for queued requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from app.core.errors import GenerationDomainError

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None


class JobQueue:
    """Owns the single background worker and the in-memory job table."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._queue: Optional["asyncio.Queue[tuple[str, Callable[[], Any]]]"] = None
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Starts the background worker. Called once at app startup
        (see main.py's lifespan).

        The ``asyncio.Queue`` is deliberately created here, not in
        ``__init__`` - asyncio queues/locks bind to whichever event loop is
        running the first time they're actually used, and raise
        "bound to a different event loop" if reused from a different loop
        later. Creating it inside ``start()`` guarantees it's always bound
        to the loop that's about to run it (matters for tests that spin up
        a fresh event loop per app instance; a real single long-running
        process only ever calls ``start()`` once anyway).
        """
        if self._worker_task is None:
            self._queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancels the background worker and drops the (loop-bound) queue.
        Called at app shutdown. In-memory job state is simply dropped -
        nothing to persist by design."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None
        self._queue = None

    def create(self) -> Job:
        """Registers a new job (status=queued) without enqueueing work yet,
        so a caller can use the assigned id (e.g. to name a saved file)
        before handing off the actual work callable."""
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job
        return job

    def enqueue(self, job_id: str, fn: Callable[[], Any]) -> None:
        """Queues ``fn`` (a plain blocking callable, e.g. a closure around
        ``pipeline.redesign_room``) to run for the given job. ``fn`` runs in
        a worker thread (``asyncio.to_thread``) so the event loop stays
        responsive for status polling while generation runs."""
        if self._queue is None:
            raise RuntimeError("JobQueue.start() must run (app startup/lifespan) before enqueueing jobs.")
        self._queue.put_nowait((job_id, fn))

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def _run(self) -> None:
        while True:
            job_id, fn = await self._queue.get()
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            try:
                job.result = await asyncio.to_thread(fn)
                job.status = JobStatus.SUCCEEDED
            except GenerationDomainError as exc:
                logger.exception("Generation job %s failed", job_id)
                job.error = exc.user_message
                job.status = JobStatus.FAILED
            except Exception:  # noqa: BLE001 - a failed job must not crash the worker
                logger.exception("Unexpected failure in generation job %s", job_id)
                job.error = "Generation failed unexpectedly. Please try again."
                job.status = JobStatus.FAILED
            finally:
                job.finished_at = time.time()
                self._queue.task_done()


job_queue = JobQueue()
