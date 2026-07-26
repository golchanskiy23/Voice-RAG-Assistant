from app.adapters.llm.base import LLMClient
from app.adapters.llm.stub import StubLLMClient
from app.adapters.registry import Registry

llm_registry: Registry[LLMClient] = Registry("llm")
llm_registry.register("stub", StubLLMClient)
