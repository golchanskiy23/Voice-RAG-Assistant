import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import replace

import pytest

from app.adapters.embedders.hashing import HashingEmbedder
from app.adapters.vector_stores.base import SearchHit
from app.core.errors import AppError
from app.services.fusion import reciprocal_rank_fusion
from app.services.search import SearchDefaults, SearchService
from tests.llm_fakes import FixedStore, hit

A, B, C, D = (hit(name) for name in "ABCD")


def test_rrf_rewards_agreement_over_single_top_rank() -> None:
    fused = reciprocal_rank_fusion({"vector": [A, B, C], "bm25": [D, B, C]}, k=60)

    assert [h.text for h in fused] == ["B", "C", "A", "D"]
    assert fused[0].found_by == ("vector", "bm25")
    assert fused[3].found_by == ("bm25",)
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 62)


def test_rrf_ignores_raw_score_scale() -> None:
    loud = [replace(A, score=1000.0), replace(B, score=999.0)]
    quiet = [replace(B, score=0.2), replace(A, score=0.1)]

    fused = reciprocal_rank_fusion({"bm25": loud, "vector": quiet})

    assert fused[0].score == fused[1].score


class FixedKeyword:
    def __init__(self, hits: Sequence[SearchHit]) -> None:
        self.hits = [replace(h, found_by=("bm25",)) for h in hits]
        self.calls: list[int] = []

    async def search(
        self, query: str, k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]:
        self.calls.append(k)
        return self.hits[:k]


class ReverseReranker:
    name = "reverse"

    def __init__(self) -> None:
        self.seen: list[int] = []

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        self.seen.append(len(hits))
        return [replace(h, score=float(i)) for i, h in enumerate(reversed(hits))]

    async def is_ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def service(
    reranker: ReverseReranker | None = None, keyword: FixedKeyword | None = None
) -> SearchService:
    return SearchService(
        HashingEmbedder(),
        {"qdrant": FixedStore([A, B, C])},
        SearchDefaults(mode="vector", rerank_candidates=3),
        keyword if keyword is not None else FixedKeyword([D, B]),
        reranker,
    )


def test_modes_select_retrievers() -> None:
    svc = service()

    vector = asyncio.run(svc.search("q", 2, mode="vector"))
    bm25 = asyncio.run(svc.search("q", 2, mode="bm25"))
    hybrid = asyncio.run(svc.search("q", 4, mode="hybrid"))

    assert [h.text for h in vector.hits] == ["A", "B"]
    assert set(vector.latency_ms) == {"embed", "search"}
    assert [h.text for h in bm25.hits] == ["D", "B"]
    assert set(bm25.latency_ms) == {"bm25"}
    assert hybrid.hits[0].text == "B"
    assert {"embed", "search", "bm25"} <= set(hybrid.latency_ms)


def test_reranker_sees_candidates_and_cuts_to_k() -> None:
    reranker = ReverseReranker()

    result = asyncio.run(service(reranker).search("q", 1))

    assert reranker.seen == [3]
    assert [h.text for h in result.hits] == ["C"]
    assert result.reranked
    assert "rerank" in result.latency_ms


def test_rerank_can_be_disabled_per_request() -> None:
    reranker = ReverseReranker()

    result = asyncio.run(service(reranker).search("q", 1, rerank=False))

    assert reranker.seen == []
    assert [h.text for h in result.hits] == ["A"]


def test_rerank_without_reranker_is_an_error() -> None:
    with pytest.raises(AppError, match="Reranker"):
        asyncio.run(service().search("q", 1, rerank=True))
