import asyncio
import importlib.util
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.embedders.base import EMBEDDING_DIM, EmbeddingDimensionError, ensure_dimension
from app.adapters.embedders.hashing import HashingEmbedder
from app.adapters.embedders.registry import embedder_registry
from app.adapters.embedders.vectors import cosine, l2_normalize
from tests.factories import build_app

MIGRATION = (
    Path(__file__).resolve().parents[1] / "migrations" / "versions" / "0001_initial_schema.py"
)


def embed_query(text: str) -> list[float]:
    return asyncio.run(HashingEmbedder().embed_query(text))


def test_vectors_have_configured_dimension_and_unit_norm() -> None:
    vectors = asyncio.run(HashingEmbedder().embed_documents(["первый документ", "второй"]))

    assert [len(v) for v in vectors] == [EMBEDDING_DIM, EMBEDDING_DIM]
    assert all(math.isclose(math.sqrt(sum(x * x for x in v)), 1.0) for v in vectors)


def test_embedding_is_deterministic_and_case_insensitive() -> None:
    assert embed_query("Векторный ПОИСК") == embed_query("векторный поиск")


def test_shared_words_increase_similarity() -> None:
    query = embed_query("проверка базы данных")
    related = embed_query("эндпоинт ready проверяет базы данных")
    unrelated = embed_query("голосовой ответ синтезируется по предложениям")

    assert cosine(query, related) > cosine(query, unrelated)


def test_query_and_document_paths_agree() -> None:
    embedder = HashingEmbedder()
    text = "одинаковый текст"

    assert asyncio.run(embedder.embed_documents([text]))[0] == asyncio.run(
        embedder.embed_query(text)
    )


def test_count_tokens_and_truncation() -> None:
    embedder = HashingEmbedder(max_tokens=3)

    assert embedder.count_tokens("раз, два; три четыре") == 4
    assert asyncio.run(embedder.embed_query("раз два три четыре")) == asyncio.run(
        embedder.embed_query("раз два три")
    )


def test_empty_text_gives_zero_vector() -> None:
    assert embed_query("") == [0.0] * EMBEDDING_DIM


def test_l2_normalize() -> None:
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_registry_creates_hashing_embedder() -> None:
    assert isinstance(embedder_registry.create("hashing"), HashingEmbedder)


def test_ensure_dimension_rejects_mismatch() -> None:
    with pytest.raises(EmbeddingDimensionError, match=r"размерности 384.*vector\(1024\)"):
        ensure_dimension(HashingEmbedder(dim=384))


def test_app_refuses_to_start_with_wrong_dimension() -> None:
    app = build_app(embedder_factory=lambda: HashingEmbedder(dim=384))

    with pytest.raises(EmbeddingDimensionError), TestClient(app):
        pass


def test_migration_uses_same_dimension() -> None:
    spec = importlib.util.spec_from_file_location("initial_schema", MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.EMBEDDING_DIM == EMBEDDING_DIM
