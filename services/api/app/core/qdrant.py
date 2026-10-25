from qdrant_client import AsyncQdrantClient


class Qdrant:
    def __init__(self, url: str, check_compatibility: bool = True) -> None:
        self.client = AsyncQdrantClient(url=url, check_compatibility=check_compatibility)

    async def is_ready(self) -> bool:
        await self.client.get_collections()
        return True

    async def close(self) -> None:
        await self.client.close()
