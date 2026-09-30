import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.adapters.base import Resource
from app.adapters.embedders.base import Embedder, ensure_chunk_fits, ensure_dimension
from app.adapters.embedders.registry import build_embedder
from app.adapters.llm.base import LLMClient
from app.adapters.llm.registry import build_llm
from app.adapters.vector_stores.base import StoreName, VectorStore
from app.adapters.vector_stores.pgvector import PgVectorStore
from app.adapters.vector_stores.qdrant import QdrantVectorStore
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.database import Database
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.qdrant import Qdrant
from app.core.request_id import request_id_middleware
from app.core.tokens import count_llm_tokens
from app.services.prompts import load_answer_prompt

STARTUP_TIMEOUT_SECONDS = 3.0


def create_app(
    settings: Settings | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
    database_factory: Callable[[Settings], Database] | None = None,
    qdrant_factory: Callable[[Settings], Qdrant] | None = None,
    embedder_factory: Callable[[], Embedder] | None = None,
    count_tokens: Callable[[str], int] = count_llm_tokens,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        configure_logging(resolved.log_level)
        prompt = load_answer_prompt(resolved.prompts_dir, resolved.prompt_version)
        embedder = embedder_factory() if embedder_factory else build_embedder(resolved)
        ensure_dimension(embedder)
        ensure_chunk_fits(embedder, resolved.chunk_size)
        llm = llm_factory() if llm_factory else build_llm(resolved)
        database = (
            database_factory(resolved) if database_factory else Database(str(resolved.database_url))
        )
        qdrant = qdrant_factory(resolved) if qdrant_factory else Qdrant(str(resolved.qdrant_url))
        qdrant_store = QdrantVectorStore(qdrant.client, f"chunks_{embedder.name}", embedder.dim)
        vector_stores: dict[StoreName, VectorStore] = {
            "pgvector": PgVectorStore(database),
            "qdrant": qdrant_store,
        }
        resources: dict[str, Resource] = {
            "postgres": database,
            "qdrant": qdrant,
            "embedder": embedder,
            "llm": llm,
        }
        app.state.settings = resolved
        app.state.llm = llm
        app.state.embedder = embedder
        app.state.database = database
        app.state.qdrant = qdrant
        app.state.resources = resources
        app.state.vector_stores = vector_stores
        app.state.index = qdrant_store
        app.state.prompt = prompt
        app.state.count_tokens = count_tokens
        try:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(STARTUP_TIMEOUT_SECONDS):
                    await qdrant_store.ensure_collection()
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
