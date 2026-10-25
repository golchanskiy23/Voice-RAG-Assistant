import asyncio
import random
from collections.abc import Awaitable, Callable


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    retry_on: tuple[type[BaseException], ...],
    attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except retry_on:
            if attempt == attempts:
                raise
            await sleep(random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1))))
    raise AssertionError("unreachable")
