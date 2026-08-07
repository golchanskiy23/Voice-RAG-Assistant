from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.base import Resource
from app.adapters.llm.base import LLMClient
from app.adapters.llm.registry import llm_registry
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.database import Database
from app.core.errors import register_error_handlers
from app.core.request_id import request_id_middleware


def create_app(
    settings: Settings | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
    database_factory: Callable[[Settings], Database] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        llm = llm_factory() if llm_factory else llm_registry.create(resolved.llm_provider)
        database = (
            database_factory(resolved) if database_factory else Database(str(resolved.database_url))
        )
        resources: dict[str, Resource] = {"postgres": database, "llm": llm}
        app.state.settings = resolved
        app.state.llm = llm
        app.state.database = database
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
