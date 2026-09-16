from functools import lru_cache
from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, PostgresDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: PostgresDsn


class Settings(DatabaseSettings):
    llm_provider: Literal["stub"] = "stub"
    embedder: Literal["hashing", "openai"] = "hashing"
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(default=128, gt=0, le=2048)
    embedding_cache_size: int = Field(default=10_000, ge=0)
    vector_store: Literal["pgvector", "qdrant"] = "qdrant"
    qdrant_url: AnyHttpUrl
    admin_api_key: SecretStr
    chunk_size: int = Field(default=300, gt=0)
    chunk_overlap: int = Field(default=45, ge=0)

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        return self

    @model_validator(mode="after")
    def openai_requires_key(self) -> Self:
        if self.embedder == "openai" and self.openai_api_key is None:
            raise ValueError("для EMBEDDER=openai нужен OPENAI_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
