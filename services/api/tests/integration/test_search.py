import asyncio
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.adapters.embedders.hashing import HashingEmbedder
from app.adapters.vector_stores.pgvector import PgVectorStore
from app.api.security import API_KEY_HEADER
from app.core.config import Settings
from app.core.database import Database
from app.main import create_app
from tests.factories import TEST_ADMIN_API_KEY

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[4]
CORPUS = {
    "spec.md": (ROOT / "docs" / "spec.md").read_bytes(),
    "README.md": (ROOT / "README.md").read_bytes(),
    "regulation.pdf": (
        Path(__file__).resolve().parents[1] / "fixtures" / "regulation_ru.pdf"
    ).read_bytes(),
}
JUNK_DOCUMENT = uuid.UUID("00000000-0000-0000-0000-000000000001")
QUERIES = [
    "как загрузить документ и какой лимит размера файла",
    "голосовой ответ задержка до первого звука",
    "гибридный поиск bm25 и векторы через rrf",
    "кто отвечает за актуальность документов",
]


@pytest.fixture(autouse=True)
def clean(clean_stores: None) -> None:
    return None


@pytest.fixture
def indexed(settings: Settings) -> Iterator[tuple[TestClient, dict[str, str]]]:
    with TestClient(create_app(settings=settings)) as client:
        ids = {}
        for name, content in CORPUS.items():
            response: httpx2.Response = client.post(
                "/documents",
                files={"file": (name, content)},
                headers={API_KEY_HEADER: TEST_ADMIN_API_KEY},
            )
            assert response.status_code == 201, response.text
            ids[name] = response.json()["id"]
        yield client, ids


def search(client: TestClient, **payload: object) -> list[dict[str, object]]:
    response: httpx2.Response = client.post("/search", json=payload)
    assert response.status_code == 200, response.text
    chunks: list[dict[str, object]] = response.json()["chunks"]
    return chunks


@pytest.mark.parametrize("query", QUERIES)
def test_both_stores_return_same_top5(
    indexed: tuple[TestClient, dict[str, str]], query: str
) -> None:
    client, _ = indexed

    pg = search(client, query=query, k=5, store="pgvector")
    qd = search(client, query=query, k=5, store="qdrant")

    pg_scores = [float(str(c["score"])) for c in pg]
    qd_scores = [float(str(c["score"])) for c in qd]

    assert len(pg) == len(qd) == 5
    assert all(abs(a - b) < 1e-4 for a, b in zip(pg_scores, qd_scores, strict=True))
    assert len({c["id"] for c in pg} & {c["id"] for c in qd}) >= 4


@pytest.mark.parametrize("store", ["pgvector", "qdrant"])
def test_filter_by_document_in_both_stores(
    indexed: tuple[TestClient, dict[str, str]], store: str
) -> None:
    client, ids = indexed

    chunks = search(client, query=QUERIES[0], k=5, store=store, doc_ids=[ids["regulation.pdf"]])

    assert chunks and {c["document_id"] for c in chunks} == {ids["regulation.pdf"]}
    assert all(c["filename"] == "regulation.pdf" and c["page"] in (1, 2) for c in chunks)


def test_pgvector_query_uses_hnsw_index(
    settings: Settings, indexed: tuple[TestClient, dict[str, str]]
) -> None:
    vector = asyncio.run(HashingEmbedder().embed_query(QUERIES[0]))

    async def plan() -> str:
        engine = create_async_engine(str(settings.database_url))
        async with engine.connect() as connection:
            await connection.execute(text("SET enable_seqscan = off"))
            rows = await connection.execute(
                text(
                    "EXPLAIN SELECT id FROM chunks "
                    "ORDER BY embedding <=> CAST(:v AS vector) LIMIT 5"
                ),
                {"v": str(vector)},
            )
            result = "\n".join(r[0] for r in rows)
        await engine.dispose()
        return result

    assert "chunks_embedding_hnsw" in asyncio.run(plan())


def test_pgvector_store_scores_are_cosine_similarity(
    settings: Settings, indexed: tuple[TestClient, dict[str, str]]
) -> None:
    async def run() -> list[float]:
        database = Database(str(settings.database_url))
        hits = await PgVectorStore(database).search(
            await HashingEmbedder().embed_query(QUERIES[1]), k=5
        )
        await database.close()
        return [h.score for h in hits]

    scores = asyncio.run(run())

    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_unknown_document_filter_returns_nothing(
    indexed: tuple[TestClient, dict[str, str]],
) -> None:
    client, _ = indexed

    for store in ("pgvector", "qdrant"):
        assert search(client, query="документ", k=5, store=store, doc_ids=[str(uuid.uuid4())]) == []


def test_pgvector_returns_k_results_despite_dead_index_entries(
    settings: Settings, indexed: tuple[TestClient, dict[str, str]]
) -> None:
    url = str(settings.database_url)
    database_name = url.rsplit("/", 1)[-1]
    query = asyncio.run(HashingEmbedder().embed_query(QUERIES[0]))

    async def run() -> int:
        engine = create_async_engine(url)
        async with engine.begin() as connection:
            await connection.execute(text("ALTER TABLE chunks SET (autovacuum_enabled = false)"))
            await connection.execute(
                text(f"ALTER DATABASE {database_name} SET enable_seqscan = off")
            )
            await connection.execute(
                text(
                    "INSERT INTO documents (id, filename, sha256, mime) "
                    "VALUES (:id, 'junk.md', repeat('f', 64), 'text/markdown')"
                ),
                {"id": JUNK_DOCUMENT},
            )
            await connection.execute(
                text(
                    "INSERT INTO chunks (document_id, ord, text, tokens, embedding) "
                    "SELECT :id, g, 'junk', 1, CAST(:v AS vector) FROM generate_series(0, 1999) g"
                ),
                {"id": JUNK_DOCUMENT, "v": str(query)},
            )
            await connection.execute(text("DELETE FROM documents WHERE filename = 'junk.md'"))
        database = Database(url)
        try:
            return len(await PgVectorStore(database).search(query, k=5))
        finally:
            await database.close()
            async with engine.begin() as connection:
                await connection.execute(
                    text(f"ALTER DATABASE {database_name} RESET enable_seqscan")
                )
                await connection.execute(text("ALTER TABLE chunks RESET (autovacuum_enabled)"))
            await engine.dispose()

    assert asyncio.run(run()) == 5
