from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.deps import DialogServiceDep, RAGServiceDep
from app.api.schemas import AskRequest, AskResponse, SourceOut
from app.api.sse import SSE_HEADERS, stream_sse

router = APIRouter(tags=["rag"])


@router.post("/ask")
async def ask(payload: AskRequest, rag: RAGServiceDep, dialog: DialogServiceDep) -> AskResponse:
    if payload.session_id:
        result = await dialog.answer(f"api:{payload.session_id}", payload.question)
        answer = result.answer
    else:
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


@router.post("/ask/stream", response_class=StreamingResponse)
async def ask_stream(
    payload: AskRequest, rag: RAGServiceDep, request: Request
) -> StreamingResponse:
    return StreamingResponse(
        stream_sse(request, rag.stream(payload.question), request.state.request_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
