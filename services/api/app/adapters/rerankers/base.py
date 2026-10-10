from collections.abc import Sequence
from typing import Protocol

from app.adapters.base import Resource
from app.adapters.vector_stores.base import SearchHit


class Reranker(Resource, Protocol):
    @property
    def name(self) -> str: ...

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]: ...
