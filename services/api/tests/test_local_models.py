import asyncio

import pytest

pytest.importorskip("torch")

import torch

from app.adapters.embedders.pooling import (
    cls_pool,
    mean_pool,
    normalize_rows,
    pad_columns,
)
from app.adapters.embedders.transformers import (
    BGE_M3,
    E5_BASE,
    TransformersEmbedder,
)
from app.adapters.rerankers.cross_encoder import CrossEncoderReranker
from tests.llm_fakes import hit


def test_mean_pool_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])

    assert mean_pool(hidden, mask).tolist() == [[2.0, 2.0]]
    assert cls_pool(hidden, mask).tolist() == [[1.0, 1.0]]


def test_zero_padding_keeps_cosine() -> None:
    vectors = normalize_rows(torch.randn(2, 768))
    padded = pad_columns(vectors, 1024)

    assert padded.shape == (2, 1024)
    assert torch.allclose(padded[0] @ padded[1], vectors[0] @ vectors[1])
    assert torch.allclose(padded.norm(dim=1), torch.ones(2))


class FakeEncoder:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode(self, texts: list[str]) -> torch.Tensor:
        self.batches.append(texts)
        return torch.stack([torch.full((self.dim,), float(len(t))) for t in texts])


def test_e5_adds_prefixes_batches_and_pads() -> None:
    encoder = FakeEncoder(768)
    embedder = TransformersEmbedder(E5_BASE, encoder, batch_size=2)

    documents = asyncio.run(embedder.embed_documents(["a", "b", "c"]))
    query = asyncio.run(embedder.embed_query("q"))

    assert [len(b) for b in encoder.batches] == [2, 1, 1]
    assert encoder.batches[0] == ["passage: a", "passage: b"]
    assert encoder.batches[-1] == ["query: q"]
    assert len(documents[0]) == len(query) == 1024
    assert sum(x * x for x in query) == pytest.approx(1.0)
    assert embedder.count_tokens("два слова") == 3


def test_bge_m3_has_no_prefixes() -> None:
    encoder = FakeEncoder(1024)

    asyncio.run(TransformersEmbedder(BGE_M3, encoder).embed_query("вопрос"))

    assert encoder.batches == [["вопрос"]]


class LengthScorer:
    def __init__(self) -> None:
        self.batches: list[int] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.batches.append(len(passages))
        return [len(p) / 100 for p in passages]


def test_cross_encoder_sorts_by_pair_score_in_batches() -> None:
    scorer = LengthScorer()
    hits = [hit("коротко"), hit("подлиннее текст"), hit("самый длинный текст из всех")]

    ranked = asyncio.run(CrossEncoderReranker(scorer, batch_size=2).rerank("q", hits))

    assert [h.text for h in ranked] == [hits[2].text, hits[1].text, hits[0].text]
    assert scorer.batches == [2, 1]
    assert ranked[0].score == pytest.approx(len(hits[2].text) / 100)
