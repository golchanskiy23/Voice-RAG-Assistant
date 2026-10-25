from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.base import Resource
from app.adapters.llm.base import LLMClient
from app.core.config import Settings
from app.core.database import Database
from app.repositories.documents import DocumentRepository, SqlDocumentRepository
from app.services.ingestion import IngestionService
from app.services.rag import RAGService


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_llm(request: Request) -> LLMClient:
    llm: LLMClient = request.app.state.llm
    return llm


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


def get_ingestion_service(
    documents: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> IngestionService:
    return IngestionService(documents)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
ResourcesDep = Annotated[dict[str, Resource], Depends(get_resources)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
