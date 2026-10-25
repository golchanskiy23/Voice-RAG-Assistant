from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import Resource
from app.adapters.embedders.base import Embedder
from app.adapters.llm.base import LLMClient
from app.adapters.vector_stores.base import StoreName, VectorStore
from app.core.config import Settings
from app.core.database import Database
from app.repositories.documents import DocumentRepository, SqlDocumentRepository
from app.services.chunking import Chunker
from app.services.ingestion import IngestionService
from app.services.rag import RAGService
from app.services.search import SearchService


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


def get_rag_service(llm: Annotated[LLMClient, Depends(get_llm)]) -> RAGService:
    return RAGService(llm)


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


def get_search_service(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    stores: Annotated[dict[StoreName, VectorStore], Depends(get_vector_stores)],
) -> SearchService:
    return SearchService(embedder, stores, settings.vector_store)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ResourcesDep = Annotated[dict[str, Resource], Depends(get_resources)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
