import hashlib
import re
from collections.abc import Sequence

from app.adapters.embedders.base import EMBEDDING_DIM
from app.adapters.embedders.vectors import l2_normalize

WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


class HashingEmbedder:
    def __init__(self, dim: int = EMBEDDING_DIM, max_tokens: int = 512) -> None:
        self._dim = dim
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        return len(words(text))

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def is_ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in words(text)[: self._max_tokens]:
            digest = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            vector[digest % self._dim] += 1.0 if digest >> 63 else -1.0
        return l2_normalize(vector)
