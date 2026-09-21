import pytest

from app.adapters.llm.registry import llm_registry
from app.adapters.llm.stub import StubLLMClient
from app.adapters.registry import Registry, UnknownAdapterError
from tests.factories import make_settings


def test_creates_registered_implementation_with_context() -> None:
    registry: Registry[float, StubLLMClient] = Registry("llm")
    registry.register("stub", lambda delay: StubLLMClient(token_delay=delay))

    client = registry.create("stub", 0.5)

    assert isinstance(client, StubLLMClient)
    assert client._token_delay == 0.5


def test_each_create_returns_new_instance() -> None:
    settings = make_settings()

    assert llm_registry.create("stub", settings) is not llm_registry.create("stub", settings)


def test_unknown_name_lists_available() -> None:
    with pytest.raises(UnknownAdapterError, match=r"llm: неизвестная реализация 'gpt'.*\['stub'\]"):
        llm_registry.create("gpt", make_settings())


def test_duplicate_registration_is_rejected() -> None:
    registry: Registry[None, StubLLMClient] = Registry("llm")
    registry.register("stub", lambda _: StubLLMClient())

    with pytest.raises(ValueError, match="уже зарегистрирована"):
        registry.register("stub", lambda _: StubLLMClient())
