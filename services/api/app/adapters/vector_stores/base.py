import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

StoreName = Literal["pgvector", "qdrant"]


@dataclass(frozen=True)
class VectorPoint:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    ord: int
    text: str
    page: int | None
    section: str | None
    vector: list[float]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    ord: int
    text: str
    page: int | None
    section: str | None
    score: float


class VectorStore(Protocol):
    @property
    def name(self) -> StoreName: ...

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    async def search(
        self, vector: list[float], k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]: ...

    async def delete_documents(self, document_ids: Sequence[uuid.UUID]) -> None: ...
