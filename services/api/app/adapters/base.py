from typing import Protocol


class Resource(Protocol):
    async def is_ready(self) -> bool: ...

    async def close(self) -> None: ...
