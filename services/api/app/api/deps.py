from typing import Annotated

from fastapi import Depends, Request

from app.adapters.base import Resource
from app.adapters.llm.base import LLMClient
from app.services.rag import RAGService


def get_llm(request: Request) -> LLMClient:
    llm: LLMClient = request.app.state.llm
    return llm


def get_resources(request: Request) -> dict[str, Resource]:
    resources: dict[str, Resource] = request.app.state.resources
    return resources


def get_rag_service(llm: Annotated[LLMClient, Depends(get_llm)]) -> RAGService:
    return RAGService(llm)


LLMDep = Annotated[LLMClient, Depends(get_llm)]
ResourcesDep = Annotated[dict[str, Resource], Depends(get_resources)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
