from fastapi import APIRouter

from app.api.deps import SearchServiceDep
from app.api.schemas import SearchChunkOut, SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search")
async def search(payload: SearchRequest, service: SearchServiceDep) -> SearchResponse:
    result = await service.search(payload.query, payload.k, payload.doc_ids, payload.store)
    return SearchResponse(
        chunks=[
            SearchChunkOut(
                id=hit.chunk_id,
                document_id=hit.document_id,
                filename=hit.filename,
                ord=hit.ord,
                text=hit.text,
                page=hit.page,
                section=hit.section,
                score=round(hit.score, 6),
                source="vector",
            )
            for hit in result.hits
        ],
        store=result.store,
        latency=result.latency_ms,
    )
