from fastapi import APIRouter

from app.api.deps import RAGServiceDep
from app.api.schemas import AskRequest, AskResponse, SourceOut

router = APIRouter(tags=["rag"])


@router.post("/ask")
async def ask(payload: AskRequest, rag: RAGServiceDep) -> AskResponse:
    answer = await rag.answer(payload.question)
    return AskResponse(
        answer=answer.text,
        sources=[
            SourceOut(filename=s.filename, page=s.page, section=s.section, quote=s.quote)
            for s in answer.sources
        ],
        confidence=answer.confidence,
        latency=answer.latency_ms,
    )
