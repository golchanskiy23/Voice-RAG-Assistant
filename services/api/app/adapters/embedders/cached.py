import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from typing import Literal, Protocol

from app.adapters.embedders.base import Embedder

Kind = Literal["query", "document"]


class EmbeddingCache(Protocol):
    def get(self, key: str) -> list[float] | None: ...

    def set(self, key: str, vector: list[float]) -> None: ...


class LRUEmbeddingCache:
    def __init__(self, max_items: int) -> None:
        self._max_items = max_items
        self._items: OrderedDict[str, list[float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: str) -> list[float] | None:
        vector = self._items.get(key)
        if vector is not None:
            self._items.move_to_end(key)
        return vector

    def set(self, key: str, vector: list[float]) -> None:
        self._items[key] = vector
        self._items.move_to_end(key)
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def max_tokens(self) -> int:
        return self._inner.max_tokens

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        cached = [self._cache.get(self._key("document", text)) for text in texts]
        missing = list(dict.fromkeys(t for t, v in zip(texts, cached, strict=True) if v is None))
        fresh: dict[str, list[float]] = {}
        if missing:
            vectors = await self._inner.embed_documents(missing)
            for text, vector in zip(missing, vectors, strict=True):
                fresh[text] = vector
                self._cache.set(self._key("document", text), vector)
        return [v if v is not None else fresh[t] for t, v in zip(texts, cached, strict=True)]

    async def embed_query(self, text: str) -> list[float]:
        key = self._key("query", text)
        vector = self._cache.get(key)
        if vector is None:
            vector = await self._inner.embed_query(text)
            self._cache.set(key, vector)
        return vector

    async def is_ready(self) -> bool:
        return await self._inner.is_ready()

    async def close(self) -> None:
        await self._inner.close()

    def _key(self, kind: Kind, text: str) -> str:
        raw = f"{self._inner.name}:{self._inner.dim}:{kind}:{text}"
        return hashlib.sha256(raw.encode()).hexdigest()
