from typing import Literal

from fastapi import APIRouter

from app.adapters.vector_stores.base import SearchHit
from app.api.deps import SearchServiceDep
from app.api.schemas import SearchChunkOut, SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


def source_label(hit: SearchHit) -> Literal["vector", "bm25", "hybrid"]:
    if len(set(hit.found_by)) > 1:
        return "hybrid"
    return hit.found_by[0]


@router.post("/search")
async def search(payload: SearchRequest, service: SearchServiceDep) -> SearchResponse:
    result = await service.search(
        payload.query, payload.k, payload.doc_ids, payload.store, payload.mode, payload.rerank
    )
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
                source=source_label(hit),
                found_by=list(hit.found_by),
            )
            for hit in result.hits
        ],
        store=result.store,
        mode=result.mode,
        reranked=result.reranked,
        latency=result.latency_ms,
    )
