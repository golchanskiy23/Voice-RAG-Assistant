from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.deps import LLMDep
from app.api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["service"])


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
async def ready(llm: LLMDep) -> JSONResponse:
    checks = {"llm": await llm.is_ready()}
    is_ready = all(checks.values())
    body = ReadyResponse(status="ready" if is_ready else "not_ready", checks=checks)
    return JSONResponse(status_code=200 if is_ready else 503, content=body.model_dump())
