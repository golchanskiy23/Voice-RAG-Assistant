import json

from redis.asyncio import Redis

from app.adapters.dialog.base import Turn


class RedisDialogStore:
    def __init__(self, client: Redis, max_turns: int = 10, ttl_seconds: int = 86_400) -> None:
        self._redis = client
        self._max_turns = max_turns
        self._ttl = ttl_seconds

    @classmethod
    def from_url(cls, url: str, max_turns: int, ttl_seconds: int) -> "RedisDialogStore":
        return cls(Redis.from_url(url, decode_responses=True), max_turns, ttl_seconds)

    async def history(self, session: str) -> list[Turn]:
        rows = await self._redis.lrange(self._history_key(session), 0, -1)
        return [Turn(**json.loads(row)) for row in rows]

    async def append(self, session: str, turns: list[Turn]) -> None:
        key = self._history_key(session)
        rows = [json.dumps({"role": t.role, "text": t.text}, ensure_ascii=False) for t in turns]
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *rows)
            pipe.ltrim(key, -self._max_turns, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def clear(self, session: str) -> None:
        await self._redis.delete(self._history_key(session))

    async def first_seen(self, key: str, ttl_seconds: int) -> bool:
        return bool(await self._redis.set(f"seen:{key}", "1", nx=True, ex=ttl_seconds))

    async def hit_rate_limit(self, key: str, limit: int, window_seconds: int) -> bool:
        redis_key = f"rate:{key}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(redis_key)
            pipe.expire(redis_key, window_seconds, nx=True)
            count, _ = await pipe.execute()
        return int(count) > limit

    async def is_ready(self) -> bool:
        return bool(await self._redis.ping())

    async def close(self) -> None:
        await self._redis.aclose()

    @staticmethod
    def _history_key(session: str) -> str:
        return f"dialog:{session}"
