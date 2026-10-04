"""Production API router aggregation."""

from fastapi import APIRouter

from app.api import (
    routes_edit,
    routes_generate,
    routes_history,
    routes_jobs,
    routes_reference,
)

api_router = APIRouter()
api_router.include_router(routes_generate.router)
api_router.include_router(routes_edit.router)
api_router.include_router(routes_reference.router)
api_router.include_router(routes_history.router)
api_router.include_router(routes_jobs.router)
