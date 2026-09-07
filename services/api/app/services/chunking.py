import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from app.parsing.models import Block

SECTION_SEPARATOR = " > "
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[«\"(]?[A-ZА-ЯЁ0-9])")
ABBREVIATIONS = (
    "т.е.",
    "т.д.",
    "т.п.",
    "т. е.",
    "т. д.",
    "т. п.",
    "г.",
    "гг.",
    "см.",
    "e.g.",
    "i.e.",
)


@dataclass(frozen=True)
class ChunkDraft:
    ord: int
    text: str
    tokens: int
    section: str | None
    pages: tuple[int, ...]

    @property
    def page(self) -> int | None:
        return self.pages[0] if self.pages else None


@dataclass(frozen=True)
class Unit:
    text: str
    tokens: int
    pages: tuple[int, ...]
    starts_paragraph: bool


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for piece in SENTENCE_BOUNDARY.split(text):
        if parts and parts[-1].lower().endswith(ABBREVIATIONS):
            parts[-1] = f"{parts[-1]} {piece}"
        else:
            parts.append(piece)
    return [p.strip() for p in parts if p.strip()]


class Chunker:
    def __init__(self, count_tokens: Callable[[str], int], chunk_size: int, overlap: int) -> None:
        if not 0 <= overlap < chunk_size:
            raise ValueError("перекрытие должно быть неотрицательным и меньше размера чанка")
        self._count = count_tokens
        self._size = chunk_size
        self._overlap = overlap

    def split(self, blocks: Sequence[Block]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for section, units in self._sections(blocks):
            for text, pages in self._pack(units):
                drafts.append(
                    ChunkDraft(
                        ord=len(drafts),
                        text=text,
                        tokens=max(1, self._count(text)),
                        section=SECTION_SEPARATOR.join(section) or None,
                        pages=pages,
                    )
                )
        return drafts

    def _sections(self, blocks: Sequence[Block]) -> Iterator[tuple[tuple[str, ...], list[Unit]]]:
        section: tuple[str, ...] = ()
        units: list[Unit] = []
        for block in blocks:
            if block.kind == "heading" or block.section != section:
                if units:
                    yield section, units
                units = []
                section = block.section
            if block.kind != "heading":
                units.extend(self._units(block))
        if units:
            yield section, units

    def _units(self, block: Block) -> Iterator[Unit]:
        pieces = [block.text] if block.kind == "code" else split_sentences(block.text)
        for index, piece in enumerate(pieces):
            tokens = self._count(piece)
            if tokens <= self._size:
                yield Unit(piece, tokens, block.pages, index == 0)
                continue
            for window_index, window in enumerate(self._windows(piece)):
                yield Unit(
                    window, self._count(window), block.pages, index == 0 and window_index == 0
                )

    def _windows(self, text: str) -> Iterator[str]:
        words = text.split()
        current: list[str] = []
        for word in words:
            if current and self._count(" ".join([*current, word])) > self._size:
                yield " ".join(current)
                current = []
            current.append(word)
        if current:
            yield " ".join(current)

    def _pack(self, units: list[Unit]) -> Iterator[tuple[str, tuple[int, ...]]]:
        current: list[Unit] = []
        size = 0
        for unit in units:
            if current and size + unit.tokens > self._size:
                yield self._render(current)
                current = self._tail(current, unit.tokens)
                size = sum(u.tokens for u in current)
            current.append(unit)
            size += unit.tokens
        if current:
            yield self._render(current)

    def _tail(self, units: list[Unit], incoming: int) -> list[Unit]:
        tail: list[Unit] = []
        size = 0
        for unit in reversed(units):
            if size + unit.tokens > self._overlap or size + unit.tokens + incoming > self._size:
                break
            tail.insert(0, unit)
            size += unit.tokens
        return tail

    @staticmethod
    def _render(units: list[Unit]) -> tuple[str, tuple[int, ...]]:
        text = units[0].text
        for unit in units[1:]:
            text += ("\n\n" if unit.starts_paragraph else " ") + unit.text
        pages = tuple(dict.fromkeys(page for unit in units for page in unit.pages))
        return text, pages
