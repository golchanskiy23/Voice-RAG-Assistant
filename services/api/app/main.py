from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.llm.base import LLMClient
from app.adapters.llm.stub import StubLLMClient
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.request_id import request_id_middleware

LLM_FACTORIES: dict[str, Callable[[], LLMClient]] = {"stub": StubLLMClient}


def create_app(
    settings: Settings | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    make_llm = llm_factory or LLM_FACTORIES[settings.llm_provider]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.llm = make_llm()
        try:
            yield
        finally:
            await app.state.llm.close()

    app = FastAPI(title="Voice RAG Assistant", lifespan=lifespan)
    app.middleware("http")(request_id_middleware)
    register_error_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
