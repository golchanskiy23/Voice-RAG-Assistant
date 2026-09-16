from app.adapters.llm.base import LLMClient
from app.adapters.llm.stub import StubLLMClient
from app.adapters.registry import Registry
from app.core.config import Settings

llm_registry: Registry[Settings, LLMClient] = Registry("llm")
llm_registry.register("stub", lambda _: StubLLMClient())
