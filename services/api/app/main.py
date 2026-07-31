from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.base import Resource
from app.adapters.llm.base import LLMClient
from app.adapters.llm.registry import llm_registry
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.request_id import request_id_middleware


def create_app(
    settings: Settings | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        llm = llm_factory() if llm_factory else llm_registry.create(settings.llm_provider)
        resources: dict[str, Resource] = {"llm": llm}
        app.state.settings = settings
        app.state.llm = llm
        app.state.resources = resources
        try:
            yield
        finally:
            for resource in reversed(resources.values()):
                await resource.close()

    app = FastAPI(title="Voice RAG Assistant", lifespan=lifespan)
    app.middleware("http")(request_id_middleware)
    register_error_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
