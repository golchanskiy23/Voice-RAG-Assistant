import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from app.adapters.embedders.base import EMBEDDING_DIM
from app.adapters.embedders.pooling import cls_pool, mean_pool, normalize_rows, pad_columns

Pooling = Literal["mean", "cls"]
POOLERS: dict[Pooling, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mean": mean_pool,
    "cls": cls_pool,
}


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    model_id: str
    pooling: Pooling
    native_dim: int
    max_length: int = 512
    query_prefix: str = ""
    passage_prefix: str = ""


E5_BASE = EncoderSpec(
    name="e5",
    model_id="intfloat/multilingual-e5-base",
    pooling="mean",
    native_dim=768,
    query_prefix="query: ",
    passage_prefix="passage: ",
)
BGE_M3 = EncoderSpec(name="bge-m3", model_id="BAAI/bge-m3", pooling="cls", native_dim=1024)


class Encoder(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def encode(self, texts: list[str]) -> torch.Tensor: ...


class HFEncoder:
    def __init__(self, spec: EncoderSpec, device: str = "cpu", fp16: bool = False) -> None:
        self._spec = spec
        self._device = torch.device(device)
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(spec.model_id)
        dtype = torch.float16 if fp16 and self._device.type == "cuda" else torch.float32
        self._model = AutoModel.from_pretrained(spec.model_id, dtype=dtype).to(self._device)
        self._model.eval()

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode(self, texts: list[str]) -> torch.Tensor:
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._spec.max_length,
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            hidden = self._model(**batch).last_hidden_state
        return POOLERS[self._spec.pooling](hidden, batch["attention_mask"]).cpu()


class TransformersEmbedder:
    def __init__(
        self,
        spec: EncoderSpec,
        encoder: Encoder,
        batch_size: int = 32,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        self._spec = spec
        self._encoder = encoder
        self._batch_size = batch_size
        self._dim = dim
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> int:
        return self._spec.max_length

    def count_tokens(self, text: str) -> int:
        return self._encoder.count_tokens(self._spec.passage_prefix + text)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [self._spec.passage_prefix + text for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prefixed), self._batch_size):
            vectors.extend(await self._embed(prefixed[start : start + self._batch_size]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([self._spec.query_prefix + text]))[0]

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True

    async def _embed(self, batch: list[str]) -> list[list[float]]:
        async with self._lock:
            return await asyncio.to_thread(self._embed_sync, batch)

    def _embed_sync(self, batch: list[str]) -> list[list[float]]:
        vectors = pad_columns(normalize_rows(self._encoder.encode(batch)), self._dim)
        result: list[list[float]] = vectors.tolist()
        return result
