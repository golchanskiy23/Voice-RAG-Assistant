from collections.abc import Callable

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


def transformers_embedder(model: str) -> Callable[[Settings], Embedder]:
    def build(settings: Settings) -> Embedder:
        from app.adapters.embedders.transformers import (
            BGE_M3,
            E5_BASE,
            HFEncoder,
            TransformersEmbedder,
        )

        spec = {"e5": E5_BASE, "bge-m3": BGE_M3}[model]
        encoder = HFEncoder(spec, device=settings.embedder_device)
        return TransformersEmbedder(spec, encoder, batch_size=settings.embedding_batch_size)

    return build


embedder_registry: Registry[Settings, Embedder] = Registry("embedder")
embedder_registry.register("hashing", lambda _: HashingEmbedder())
embedder_registry.register("openai", openai_embedder)
embedder_registry.register("e5", transformers_embedder("e5"))
embedder_registry.register("bge-m3", transformers_embedder("bge-m3"))


def build_embedder(settings: Settings) -> Embedder:
    embedder = embedder_registry.create(settings.embedder, settings)
    if settings.embedding_cache_size == 0:
        return embedder
    return CachedEmbedder(embedder, LRUEmbeddingCache(settings.embedding_cache_size))
