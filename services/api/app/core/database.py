from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class Database:
    def __init__(self, url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(url, pool_pre_ping=True)
        self.sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def is_ready(self) -> bool:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True

    async def close(self) -> None:
        await self.engine.dispose()
