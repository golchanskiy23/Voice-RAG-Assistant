from app.adapters.llm.base import LLMClient
from app.adapters.llm.fallback import FallbackLLMClient
from app.adapters.llm.gigachat import GigaChatLLMClient
from app.adapters.llm.openai import OpenAILLMClient
from app.adapters.llm.stub import StubLLMClient
from app.adapters.registry import Registry
from app.core.config import Settings


def openai_llm(settings: Settings) -> OpenAILLMClient:
    if settings.openai_api_key is None:
        raise ValueError("для LLM openai нужен OPENAI_API_KEY")
    return OpenAILLMClient(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_llm_model,
        timeout=settings.llm_timeout,
    )


def gigachat_llm(settings: Settings) -> GigaChatLLMClient:
    if settings.gigachat_credentials is None:
        raise ValueError("для LLM gigachat нужен GIGACHAT_CREDENTIALS")
    return GigaChatLLMClient.from_credentials(
        credentials=settings.gigachat_credentials.get_secret_value(),
        scope=settings.gigachat_scope,
        model=settings.gigachat_model,
        timeout=settings.llm_timeout,
        verify_ssl_certs=settings.gigachat_verify_ssl,
        ca_bundle_file=settings.gigachat_ca_bundle_file,
    )


llm_registry: Registry[Settings, LLMClient] = Registry("llm")
llm_registry.register("stub", lambda s: StubLLMClient(token_delay=s.stub_llm_token_delay))
llm_registry.register("openai", openai_llm)
llm_registry.register("gigachat", gigachat_llm)


def build_llm(settings: Settings) -> LLMClient:
    primary = llm_registry.create(settings.llm_provider, settings)
    if settings.llm_fallback is None:
        return primary
    return FallbackLLMClient(primary, llm_registry.create(settings.llm_fallback, settings))
