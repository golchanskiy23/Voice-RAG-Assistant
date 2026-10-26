from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import Resource
from app.adapters.embedders.base import Embedder
from app.adapters.llm.base import LLMClient
from app.adapters.rerankers.base import Reranker
from app.adapters.vector_stores.base import StoreName, VectorStore
from app.core.config import Settings
from app.core.database import Database
from app.repositories.documents import (
    DocumentRepository,
    SqlDocumentRepository,
    StoredDocument,
)
from app.repositories.fulltext import KeywordIndex
from app.services.chunking import Chunker
from app.services.dialog import DialogService, NoRewrite, QueryRewriter
from app.services.ingestion import IngestionService, UploadResult
from app.services.prompts import AnswerPrompt
from app.services.rag import RAGService, RAGSettings
from app.services.search import SearchDefaults, SearchService


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_llm(request: Request) -> LLMClient:
    llm: LLMClient = request.app.state.llm
    return llm


def get_embedder(request: Request) -> Embedder:
    embedder: Embedder = request.app.state.embedder
    return embedder


def get_vector_stores(request: Request) -> dict[StoreName, VectorStore]:
    stores: dict[StoreName, VectorStore] = request.app.state.vector_stores
    return stores


def get_index(request: Request) -> VectorStore:
    index: VectorStore = request.app.state.index
    return index


def get_resources(request: Request) -> dict[str, Resource]:
    resources: dict[str, Resource] = request.app.state.resources
    return resources


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        yield session


def get_document_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> DocumentRepository:
    return SqlDocumentRepository(session)


def get_chunker(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> Chunker:
    return Chunker(embedder.count_tokens, settings.chunk_size, settings.chunk_overlap)


def get_ingestion_service(
    documents: Annotated[DocumentRepository, Depends(get_document_repository)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    chunker: Annotated[Chunker, Depends(get_chunker)],
    index: Annotated[VectorStore, Depends(get_index)],
) -> IngestionService:
    return IngestionService(documents, embedder, chunker, index)


def get_keyword_index(request: Request) -> KeywordIndex:
    keyword: KeywordIndex = request.app.state.keyword
    return keyword


def get_reranker(request: Request) -> Reranker | None:
    reranker: Reranker | None = request.app.state.reranker
    return reranker


def get_search_service(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    stores: Annotated[dict[StoreName, VectorStore], Depends(get_vector_stores)],
    keyword: Annotated[KeywordIndex, Depends(get_keyword_index)],
    reranker: Annotated[Reranker | None, Depends(get_reranker)],
) -> SearchService:
    defaults = SearchDefaults(
        store=settings.vector_store,
        mode=settings.search_mode,
        rerank_candidates=settings.rerank_candidates,
        rrf_k=settings.rrf_k,
    )
    return SearchService(embedder, stores, defaults, keyword, reranker)


def get_prompt(request: Request) -> AnswerPrompt:
    prompt: AnswerPrompt = request.app.state.prompt
    return prompt


def get_token_counter(request: Request) -> Callable[[str], int]:
    count_tokens: Callable[[str], int] = request.app.state.count_tokens
    return count_tokens


def get_rag_service(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    search: Annotated[SearchService, Depends(get_search_service)],
    llm: Annotated[LLMClient, Depends(get_llm)],
    prompt: Annotated[AnswerPrompt, Depends(get_prompt)],
    count_tokens: Annotated[Callable[[str], int], Depends(get_token_counter)],
) -> RAGService:
    return RAGService(
        search,
        llm,
        prompt,
        count_tokens,
        RAGSettings(
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
            context_tokens=settings.rag_context_tokens,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        ),
    )


def rag_service_from_state(state: Any) -> RAGService:
    settings: Settings = state.settings
    search = SearchService(
        state.embedder,
        state.vector_stores,
        SearchDefaults(
            store=settings.vector_store,
            mode=settings.search_mode,
            rerank_candidates=settings.rerank_candidates,
            rrf_k=settings.rrf_k,
        ),
        state.keyword,
        state.reranker,
    )
    return RAGService(
        search,
        state.llm,
        state.prompt,
        state.count_tokens,
        RAGSettings(
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
            context_tokens=settings.rag_context_tokens,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        ),
    )


def dialog_service_from_state(state: Any) -> DialogService:
    settings: Settings = state.settings
    rewriter = QueryRewriter(state.llm, state.rewrite_prompt) if settings.rewrite_enabled else None
    return DialogService(state.dialog_store, rewriter or NoRewrite(), rag_service_from_state(state))


@asynccontextmanager
async def ingestion_from_state(state: Any) -> AsyncIterator[IngestionService]:
    settings: Settings = state.settings
    database: Database = state.database
    embedder: Embedder = state.embedder
    chunker = Chunker(embedder.count_tokens, settings.chunk_size, settings.chunk_overlap)
    async with database.sessionmaker() as session:
        yield IngestionService(SqlDocumentRepository(session), embedder, chunker, state.index)


class StateLibrary:
    def __init__(self, state: Any) -> None:
        self._state = state

    async def upload(self, filename: str, content: bytes) -> UploadResult:
        async with ingestion_from_state(self._state) as service:
            return await service.upload(filename, content)

    async def list_documents(self) -> list[StoredDocument]:
        async with ingestion_from_state(self._state) as service:
            return await service.list_documents()


def get_dialog_service(request: Request) -> DialogService:
    return dialog_service_from_state(request.app.state)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ResourcesDep = Annotated[dict[str, Resource], Depends(get_resources)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
DialogServiceDep = Annotated[DialogService, Depends(get_dialog_service)]
