import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.adapters.vector_stores.base import SearchHit

BGE_RERANKER = "BAAI/bge-reranker-v2-m3"


class PairScorer(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class HFPairScorer:
    def __init__(
        self, model_id: str = BGE_RERANKER, device: str = "cpu", max_length: int = 512
    ) -> None:
        self._device = torch.device(device)
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self._device)
        self._model.eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        batch = self._tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            logits = self._model(**batch).logits.view(-1).float()
        scores: list[float] = torch.sigmoid(logits).cpu().tolist()
        return scores


class CrossEncoderReranker:
    def __init__(
        self, scorer: PairScorer, name: str = "bge-reranker", batch_size: int = 16
    ) -> None:
        self._scorer = scorer
        self._name = name
        self._batch_size = batch_size
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def name(self) -> str:
        return self._name

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        async with self._lock:
            scores = await asyncio.to_thread(self._score, query, [hit.text for hit in hits])
        rescored = [replace(hit, score=score) for hit, score in zip(hits, scores, strict=True)]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True

    def _score(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(passages), self._batch_size):
            scores.extend(self._scorer.score(query, passages[start : start + self._batch_size]))
        return scores
