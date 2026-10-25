from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["heading", "paragraph", "code"]


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str
    section: tuple[str, ...] = ()
    pages: tuple[int, ...] = ()
    level: int = 0


@dataclass(frozen=True)
class ParsedDocument:
    blocks: list[Block] = field(default_factory=list)
    pages: int | None = None


class DocumentParseError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
