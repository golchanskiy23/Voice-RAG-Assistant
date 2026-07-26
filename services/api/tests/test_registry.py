import pytest

from app.adapters.llm.registry import llm_registry
from app.adapters.llm.stub import StubLLMClient
from app.adapters.registry import Registry, UnknownAdapterError


def test_creates_registered_implementation() -> None:
    registry: Registry[StubLLMClient] = Registry("llm")
    registry.register("stub", StubLLMClient)

    assert isinstance(registry.create("stub"), StubLLMClient)


def test_each_create_returns_new_instance() -> None:
    assert llm_registry.create("stub") is not llm_registry.create("stub")


def test_unknown_name_lists_available() -> None:
    with pytest.raises(UnknownAdapterError, match=r"llm: неизвестная реализация 'gpt'.*\['stub'\]"):
        llm_registry.create("gpt")


def test_duplicate_registration_is_rejected() -> None:
    registry: Registry[StubLLMClient] = Registry("llm")
    registry.register("stub", StubLLMClient)

    with pytest.raises(ValueError, match="уже зарегистрирована"):
        registry.register("stub", StubLLMClient)
