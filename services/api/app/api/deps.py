from typing import Annotated

from fastapi import Depends, Request

from app.adapters.llm.base import LLMClient
from app.services.rag import RAGService


def get_llm(request: Request) -> LLMClient:
    llm: LLMClient = request.app.state.llm
    return llm


def get_rag_service(llm: Annotated[LLMClient, Depends(get_llm)]) -> RAGService:
    return RAGService(llm)


LLMDep = Annotated[LLMClient, Depends(get_llm)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
