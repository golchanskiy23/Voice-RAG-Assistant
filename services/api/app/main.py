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
from app.adapters.rerankers.base import Reranker
from app.adapters.rerankers.registry import build_reranker
from app.adapters.stt.base import SpeechToText
from app.adapters.tts.base import TextToSpeech
from app.adapters.vad.base import VoiceActivityDetector
from app.adapters.vector_stores.base import StoreName, VectorStore
from app.adapters.vector_stores.pgvector import PgVectorStore
from app.adapters.vector_stores.qdrant import QdrantVectorStore
from app.adapters.voice_registry import stt_registry, tts_registry, vad_factory
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.database import Database
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.qdrant import Qdrant
from app.core.request_id import request_id_middleware
from app.core.tokens import count_llm_tokens
from app.repositories.fulltext import KeywordIndex, PostgresBM25
from app.services.prompts import load_answer_prompt

STARTUP_TIMEOUT_SECONDS = 3.0


async def warm_up(*resources: object) -> None:
    for resource in resources:
        warm = getattr(resource, "warm_up", None)
        if warm is not None:
            await warm()


def create_app(
    settings: Settings | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
    database_factory: Callable[[Settings], Database] | None = None,
    qdrant_factory: Callable[[Settings], Qdrant] | None = None,
    embedder_factory: Callable[[], Embedder] | None = None,
    count_tokens: Callable[[str], int] = count_llm_tokens,
    keyword_factory: Callable[[Database], KeywordIndex] = PostgresBM25,
    reranker_factory: Callable[[], Reranker | None] | None = None,
    stt_factory: Callable[[], SpeechToText] | None = None,
    tts_factory: Callable[[], TextToSpeech] | None = None,
    vad: Callable[[], VoiceActivityDetector] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        configure_logging(resolved.log_level)
        prompt = load_answer_prompt(resolved.prompts_dir, resolved.prompt_version)
        embedder = (
            embedder_factory()
            if embedder_factory
            else await asyncio.to_thread(build_embedder, resolved)
        )
        reranker = (
            reranker_factory()
            if reranker_factory
            else await asyncio.to_thread(build_reranker, resolved)
        )
        ensure_dimension(embedder)
        ensure_chunk_fits(embedder, resolved.chunk_size)
        stt = (
            stt_factory()
            if stt_factory
            else await asyncio.to_thread(stt_registry.create, resolved.stt, resolved)
        )
        tts = (
            tts_factory()
            if tts_factory
            else await asyncio.to_thread(tts_registry.create, resolved.tts, resolved)
        )
        if resolved.voice_warm_up:
            await warm_up(stt, tts)
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
            "stt": stt,
            "tts": tts,
        }
        if reranker is not None:
            resources["reranker"] = reranker
        app.state.settings = resolved
        app.state.llm = llm
        app.state.embedder = embedder
        app.state.database = database
        app.state.qdrant = qdrant
        app.state.resources = resources
        app.state.vector_stores = vector_stores
        app.state.index = qdrant_store
        app.state.keyword = keyword_factory(database)
        app.state.reranker = reranker
        app.state.stt = stt
        app.state.tts = tts
        app.state.vad_factory = vad or vad_factory(resolved)
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
