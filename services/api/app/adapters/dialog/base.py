from dataclasses import dataclass
from typing import Literal, Protocol

from app.adapters.base import Resource


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    text: str


class DialogStore(Resource, Protocol):
    async def history(self, session: str) -> list[Turn]: ...

    async def append(self, session: str, turns: list[Turn]) -> None: ...

    async def clear(self, session: str) -> None: ...

    async def first_seen(self, key: str, ttl_seconds: int) -> bool: ...

    async def hit_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool: ...
