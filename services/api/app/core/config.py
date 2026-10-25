from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, PostgresDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: PostgresDsn


LLMProvider = Literal["stub", "openai", "gigachat"]

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class Settings(DatabaseSettings):
    llm_provider: LLMProvider = "stub"
    llm_fallback: LLMProvider | None = None
    llm_timeout: float = Field(default=30.0, gt=0)
    llm_max_tokens: int = Field(default=512, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    openai_llm_model: str = "gpt-4o-mini"
    gigachat_credentials: SecretStr | None = None
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_model: str = "GigaChat-2"
    gigachat_verify_ssl: bool = True
    gigachat_ca_bundle_file: str | None = None
    prompts_dir: Path = PROMPTS_DIR
    prompt_version: str = "v2"
    rag_top_k: int = Field(default=5, ge=1, le=50)
    rag_min_score: float = 0.0
    rag_context_tokens: int = Field(default=3000, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    stt: Literal["stub", "faster-whisper"] = "stub"
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    stt_partial_interval: float = Field(default=1.0, ge=0)
    vad: Literal["energy", "silero"] = "energy"
    vad_threshold_dbfs: float = -40.0
    vad_min_silence_ms: int = Field(default=600, ge=60)
    tts: Literal["stub", "silero"] = "stub"
    tts_speaker: str = "xenia"
    tts_sample_rate: int = 24_000
    voice_warm_up: bool = True
    voice_streaming: bool = True
    voice_first_chunk_chars: int = Field(default=80, ge=0)
    stub_llm_token_delay: float = Field(default=0.0, ge=0)
    stub_tts_realtime_factor: float = Field(default=0.0, ge=0)
    max_utterance_seconds: float = Field(default=30.0, gt=0)
    embedder: Literal["hashing", "openai", "e5", "bge-m3"] = "hashing"
    embedder_device: str = "cpu"
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(default=128, gt=0, le=2048)
    embedding_cache_size: int = Field(default=10_000, ge=0)
    vector_store: Literal["pgvector", "qdrant"] = "qdrant"
    search_mode: Literal["vector", "bm25", "hybrid"] = "bm25"
    rrf_k: int = Field(default=60, gt=0)
    reranker: Literal["none", "bge"] = "none"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    qdrant_url: AnyHttpUrl
    admin_api_key: SecretStr
    chunk_size: int = Field(default=300, gt=0)
    chunk_overlap: int = Field(default=45, ge=0)

    @field_validator(
        "llm_fallback", "gigachat_credentials", "gigachat_ca_bundle_file", mode="before"
    )
    @classmethod
    def empty_as_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        return self

    @model_validator(mode="after")
    def openai_requires_key(self) -> Self:
        uses_openai = self.embedder == "openai" or "openai" in self.llm_providers
        if uses_openai and self.openai_api_key is None:
            raise ValueError("для EMBEDDER=openai и LLM openai нужен OPENAI_API_KEY")
        return self

    @model_validator(mode="after")
    def gigachat_requires_credentials(self) -> Self:
        if "gigachat" in self.llm_providers and self.gigachat_credentials is None:
            raise ValueError("для LLM gigachat нужен GIGACHAT_CREDENTIALS")
        return self

    @model_validator(mode="after")
    def fallback_differs_from_provider(self) -> Self:
        if self.llm_fallback == self.llm_provider:
            raise ValueError("LLM_FALLBACK должен отличаться от LLM_PROVIDER")
        return self

    @property
    def llm_providers(self) -> set[LLMProvider]:
        return {self.llm_provider} | ({self.llm_fallback} if self.llm_fallback else set())


@lru_cache
def get_settings() -> Settings:
    return Settings()
