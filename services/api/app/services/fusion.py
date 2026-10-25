from collections.abc import Mapping, Sequence
from dataclasses import replace

from app.adapters.vector_stores.base import Retriever, SearchHit

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Mapping[Retriever, Sequence[SearchHit]], k: int = RRF_K
) -> list[SearchHit]:
    scores: dict[object, float] = {}
    found_by: dict[object, list[Retriever]] = {}
    hits: dict[object, SearchHit] = {}
    for retriever, ranking in rankings.items():
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            found_by.setdefault(hit.chunk_id, []).append(retriever)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [
        replace(hits[chunk_id], score=scores[chunk_id], found_by=tuple(found_by[chunk_id]))
        for chunk_id in ordered
    ]
