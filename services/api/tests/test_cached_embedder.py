import asyncio
from collections.abc import Sequence

from app.adapters.embedders.cached import CachedEmbedder, LRUEmbeddingCache
from app.adapters.embedders.hashing import HashingEmbedder


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.documents: list[list[str]] = []
        self.queries: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents.append(list(texts))
        return await super().embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return await super().embed_query(text)


def test_only_missing_documents_reach_the_model() -> None:
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, LRUEmbeddingCache(100))

    first = asyncio.run(cached.embed_documents(["а", "б"]))
    second = asyncio.run(cached.embed_documents(["б", "в", "в", "а"]))

    assert inner.documents == [["а", "б"], ["в"]]
    assert second[0] == first[1] and second[3] == first[0] and second[1] == second[2]


def test_query_cache_is_separate_from_documents() -> None:
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, LRUEmbeddingCache(100))

    asyncio.run(cached.embed_documents(["текст"]))
    asyncio.run(cached.embed_query("текст"))
    asyncio.run(cached.embed_query("текст"))

    assert inner.queries == ["текст"]


def test_batch_larger_than_cache_is_returned_in_full() -> None:
    cached = CachedEmbedder(HashingEmbedder(), LRUEmbeddingCache(2))
    texts = [f"текст {i}" for i in range(5)]

    vectors = asyncio.run(cached.embed_documents(texts))

    assert vectors == asyncio.run(HashingEmbedder().embed_documents(texts))


def test_lru_evicts_least_recently_used() -> None:
    cache = LRUEmbeddingCache(2)
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    cache.get("a")
    cache.set("c", [3.0])

    assert (cache.get("a"), cache.get("b"), cache.get("c")) == ([1.0], None, [3.0])
    assert len(cache) == 2
