from collections.abc import Sequence
from typing import Protocol

from app.adapters.base import Resource

EMBEDDING_DIM = 1024


class Embedder(Resource, Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def max_tokens(self) -> int: ...

    def count_tokens(self, text: str) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class EmbeddingDimensionError(RuntimeError):
    pass


def ensure_dimension(embedder: Embedder, expected: int = EMBEDDING_DIM) -> None:
    if embedder.dim != expected:
        raise EmbeddingDimensionError(
            f"эмбеддер {embedder.name!r} выдаёт векторы размерности {embedder.dim}, "
            f"а колонка chunks.embedding — vector({expected})"
        )
