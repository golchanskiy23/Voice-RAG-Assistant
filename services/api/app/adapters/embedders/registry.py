from app.adapters.embedders.base import Embedder
from app.adapters.embedders.cached import CachedEmbedder, LRUEmbeddingCache
from app.adapters.embedders.hashing import HashingEmbedder
from app.adapters.embedders.openai import OpenAIEmbedder
from app.adapters.registry import Registry
from app.core.config import Settings


def openai_embedder(settings: Settings) -> OpenAIEmbedder:
    if settings.openai_api_key is None:
        raise ValueError("для EMBEDDER=openai нужен OPENAI_API_KEY")
    return OpenAIEmbedder(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )


embedder_registry: Registry[Settings, Embedder] = Registry("embedder")
embedder_registry.register("hashing", lambda _: HashingEmbedder())
embedder_registry.register("openai", openai_embedder)


def build_embedder(settings: Settings) -> Embedder:
    embedder = embedder_registry.create(settings.embedder, settings)
    if settings.embedding_cache_size == 0:
        return embedder
    return CachedEmbedder(embedder, LRUEmbeddingCache(settings.embedding_cache_size))
