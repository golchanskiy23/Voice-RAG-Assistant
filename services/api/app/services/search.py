import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.adapters.embedders.base import Embedder
from app.adapters.vector_stores.base import SearchHit, StoreName, VectorStore
from app.core.errors import AppError


@dataclass(frozen=True)
class SearchResult:
    hits: list[SearchHit]
    store: StoreName
    latency_ms: dict[str, float] = field(default_factory=dict)


class SearchService:
    def __init__(
        self, embedder: Embedder, stores: Mapping[StoreName, VectorStore], default: StoreName
    ) -> None:
        self._embedder = embedder
        self._stores = stores
        self._default = default

    async def search(
        self,
        query: str,
        k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
        store: StoreName | None = None,
    ) -> SearchResult:
        name = store or self._default
        if name not in self._stores:
            raise AppError("unknown_store", f"Хранилище {name!r} не настроено", status_code=400)
        started = time.perf_counter()
        vector = await self._embedder.embed_query(query)
        embedded = time.perf_counter()
        hits = await self._stores[name].search(vector, k, document_ids)
        searched = time.perf_counter()
        return SearchResult(
            hits=hits,
            store=name,
            latency_ms={
                "embed": round((embedded - started) * 1000, 2),
                "search": round((searched - embedded) * 1000, 2),
            },
        )
