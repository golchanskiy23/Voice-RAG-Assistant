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


SPECIAL_TOKENS_RESERVE = 16


class ChunkTooLargeError(RuntimeError):
    pass


def ensure_chunk_fits(embedder: Embedder, chunk_size: int) -> None:
    if chunk_size + SPECIAL_TOKENS_RESERVE > embedder.max_tokens:
        raise ChunkTooLargeError(
            f"chunk_size={chunk_size} + {SPECIAL_TOKENS_RESERVE} служебных токенов не помещается "
            f"в лимит эмбеддера {embedder.name!r} ({embedder.max_tokens} токенов)"
        )


class EmbeddingDimensionError(RuntimeError):
    pass


def ensure_dimension(embedder: Embedder, expected: int = EMBEDDING_DIM) -> None:
    if embedder.dim != expected:
        raise EmbeddingDimensionError(
            f"эмбеддер {embedder.name!r} выдаёт векторы размерности {embedder.dim}, "
            f"а колонка chunks.embedding — vector({expected})"
        )
