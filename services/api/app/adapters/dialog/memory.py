import time
from collections import defaultdict

from app.adapters.dialog.base import Turn


class InMemoryDialogStore:
    def __init__(self, max_turns: int = 10, ttl_seconds: int = 86_400) -> None:
        self._max_turns = max_turns
        self._ttl = ttl_seconds
        self._history: dict[str, tuple[float, list[Turn]]] = {}
        self._seen: dict[str, float] = {}
        self._counters: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    async def history(self, session: str) -> list[Turn]:
        expires, turns = self._history.get(session, (0.0, []))
        return list(turns) if expires > time.monotonic() else []

    async def append(self, session: str, turns: list[Turn]) -> None:
        combined = [*(await self.history(session)), *turns][-self._max_turns :]
        self._history[session] = (time.monotonic() + self._ttl, combined)

    async def clear(self, session: str) -> None:
        self._history.pop(session, None)

    async def first_seen(self, key: str, ttl_seconds: int) -> bool:
        now = time.monotonic()
        if self._seen.get(key, 0.0) > now:
            return False
        self._seen[key] = now + ttl_seconds
        return True

    async def hit_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        reset_at, count = self._counters[key]
        if reset_at <= now:
            reset_at, count = now + window_seconds, 0
        self._counters[key] = (reset_at, count + 1)
        return count + 1 > limit

    async def is_ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None
