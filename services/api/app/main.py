from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.llm.base import LLMClient
from app.adapters.llm.stub import StubLLMClient
from app.api.router import api_router
from app.core.errors import register_error_handlers
from app.core.request_id import request_id_middleware


def create_app(llm_factory: Callable[[], LLMClient] = StubLLMClient) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.llm = llm_factory()
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
