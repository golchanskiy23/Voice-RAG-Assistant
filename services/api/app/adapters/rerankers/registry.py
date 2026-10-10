from app.adapters.registry import Registry
from app.adapters.rerankers.base import Reranker
from app.core.config import Settings


def bge_reranker(settings: Settings) -> Reranker:
    from app.adapters.rerankers.cross_encoder import CrossEncoderReranker, HFPairScorer

    return CrossEncoderReranker(HFPairScorer(settings.reranker_model, settings.reranker_device))


reranker_registry: Registry[Settings, Reranker] = Registry("reranker")
reranker_registry.register("bge", bge_reranker)


def build_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker == "none":
        return None
    return reranker_registry.create(settings.reranker, settings)
