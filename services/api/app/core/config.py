from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: PostgresDsn


class Settings(DatabaseSettings):
    llm_provider: Literal["stub"] = "stub"
    qdrant_url: AnyHttpUrl


@lru_cache
def get_settings() -> Settings:
    return Settings()
