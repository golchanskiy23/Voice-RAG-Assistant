import asyncio
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from app.adapters.embedders.base import Embedder
from app.adapters.rerankers.base import Reranker
from app.adapters.vector_stores.base import SearchHit, StoreName, VectorStore
from app.core.errors import AppError
from app.repositories.fulltext import KeywordIndex
from app.services.fusion import reciprocal_rank_fusion

SearchMode = Literal["vector", "bm25", "hybrid"]


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit]
    store: StoreName
    mode: SearchMode = "vector"
    reranked: bool = False
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchDefaults:
    store: StoreName = "qdrant"
    mode: SearchMode = "vector"
    rerank_candidates: int = 20
    rrf_k: int = 60


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class SearchService:
    def __init__(
        self,
        embedder: Embedder,
        stores: Mapping[StoreName, VectorStore],
        defaults: SearchDefaults | None = None,
        keyword: KeywordIndex | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._stores = stores
        self._defaults = defaults or SearchDefaults()
        self._keyword = keyword
        self._reranker = reranker

    async def search(
        self,
        query: str,
        k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
        store: StoreName | None = None,
        mode: SearchMode | None = None,
        rerank: bool | None = None,
    ) -> SearchResult:
        name = store or self._defaults.store
        if name not in self._stores:
            raise AppError("unknown_store", f"Хранилище {name!r} не настроено", status_code=400)
        resolved_mode = mode or self._defaults.mode
        use_reranker = self._reranker is not None if rerank is None else rerank
        if use_reranker and self._reranker is None:
            raise AppError("reranker_unavailable", "Reranker не настроен", status_code=400)

        candidates = max(k, self._defaults.rerank_candidates) if use_reranker else k
        latency: dict[str, float] = {}
        hits = await self._retrieve(query, candidates, document_ids, name, resolved_mode, latency)
        if use_reranker and self._reranker is not None and hits:
            started = time.perf_counter()
            hits = await self._reranker.rerank(query, hits)
            latency["rerank"] = elapsed_ms(started)
        return SearchResult(
            hits=hits[:k],
            store=name,
            mode=resolved_mode,
            reranked=use_reranker,
            latency_ms=latency,
        )

    async def _retrieve(
        self,
        query: str,
        n: int,
        document_ids: Sequence[uuid.UUID] | None,
        store: StoreName,
        mode: SearchMode,
        latency: dict[str, float],
    ) -> list[SearchHit]:
        if mode == "vector":
            return await self._vector(query, n, document_ids, store, latency)
        if self._keyword is None:
            raise AppError("bm25_unavailable", "Полнотекстовый поиск не настроен", status_code=400)
        if mode == "bm25":
            return await self._bm25(self._keyword, query, n, document_ids, latency)
        async with asyncio.TaskGroup() as tg:
            vector = tg.create_task(self._vector(query, n, document_ids, store, latency))
            keyword = tg.create_task(self._bm25(self._keyword, query, n, document_ids, latency))
        fused = reciprocal_rank_fusion(
            {"vector": vector.result(), "bm25": keyword.result()}, k=self._defaults.rrf_k
        )
        return fused[:n]

    async def _vector(
        self,
        query: str,
        n: int,
        document_ids: Sequence[uuid.UUID] | None,
        store: StoreName,
        latency: dict[str, float],
    ) -> list[SearchHit]:
        started = time.perf_counter()
        vector = await self._embedder.embed_query(query)
        embedded = time.perf_counter()
        hits = await self._stores[store].search(vector, n, document_ids)
        latency["embed"] = round((embedded - started) * 1000, 2)
        latency["search"] = elapsed_ms(embedded)
        return hits

    async def _bm25(
        self,
        keyword: KeywordIndex,
        query: str,
        n: int,
        document_ids: Sequence[uuid.UUID] | None,
        latency: dict[str, float],
    ) -> list[SearchHit]:
        started = time.perf_counter()
        hits = await keyword.search(query, n, document_ids)
        latency["bm25"] = elapsed_ms(started)
        return hits
