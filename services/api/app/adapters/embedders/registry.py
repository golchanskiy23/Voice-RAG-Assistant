from app.adapters.embedders.base import Embedder
from app.adapters.embedders.hashing import HashingEmbedder
from app.adapters.registry import Registry

embedder_registry: Registry[Embedder] = Registry("embedder")
embedder_registry.register("hashing", HashingEmbedder)
