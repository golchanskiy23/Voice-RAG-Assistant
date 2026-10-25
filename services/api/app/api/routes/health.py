import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.adapters.base import Resource
from app.api.deps import ResourcesDep
from app.api.schemas import HealthResponse, ReadyResponse

READY_TIMEOUT_SECONDS = 2.0

router = APIRouter(tags=["service"])


async def check_resource(resource: Resource) -> bool:
    try:
        async with asyncio.timeout(READY_TIMEOUT_SECONDS):
            return await resource.is_ready()
    except Exception:
        return False


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
async def ready(resources: ResourcesDep) -> JSONResponse:
    async with asyncio.TaskGroup() as tg:
        tasks = {name: tg.create_task(check_resource(r)) for name, r in resources.items()}
    checks = {name: task.result() for name, task in tasks.items()}
    is_ready = all(checks.values())
    body = ReadyResponse(status="ready" if is_ready else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if is_ready else 503, content=body.model_dump())
