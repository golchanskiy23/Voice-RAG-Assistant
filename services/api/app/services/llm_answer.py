import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator

from app.adapters.llm.base import JsonSchema

NO_ANSWER = "В документах нет ответа на этот вопрос."

ANSWER_SCHEMA = JsonSchema(
    name="rag_answer",
    schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "integer"}},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "sources", "confidence"],
        "additionalProperties": False,
    },
)

ANSWER_FIELD_START = re.compile(r'"answer"\s*:\s*"')
JSON_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class LLMAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    sources: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def sources_within_context(self, info: ValidationInfo) -> Self:
        chunk_count = (info.context or {}).get("chunk_count", 0)
        unknown = sorted({n for n in self.sources if not 1 <= n <= chunk_count})
        if unknown:
            raise ValueError(f"источники {unknown} вне контекста: доступны номера 1..{chunk_count}")
        return self


class AnswerValidationError(Exception):
    def __init__(self, details: str) -> None:
        super().__init__(details)
        self.details = details


def parse_llm_answer(raw: str, chunk_count: int) -> LLMAnswer:
    try:
        return LLMAnswer.model_validate_json(raw, context={"chunk_count": chunk_count})
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'answer'}: {error['msg']}"
            for error in exc.errors()
        )
        raise AnswerValidationError(details) from exc


class AnswerFieldStream:
    def __init__(self) -> None:
        self._raw: list[str] = []
        self._buffer = ""
        self._position: int | None = None
        self._finished = False

    @property
    def raw(self) -> str:
        return "".join(self._raw)

    def feed(self, delta: str) -> str:
        self._raw.append(delta)
        if self._finished:
            return ""
        self._buffer += delta
        if self._position is None:
            match = ANSWER_FIELD_START.search(self._buffer)
            if match is None:
                return ""
            self._position = match.end()
        decoded, self._position = self._decode(self._position)
        return decoded

    def _decode(self, start: int) -> tuple[str, int]:
        buffer = self._buffer
        out: list[str] = []
        i = start
        while i < len(buffer):
            char = buffer[i]
            if char == '"':
                self._finished = True
                return "".join(out), i + 1
            if char != "\\":
                out.append(char)
                i += 1
                continue
            if i + 1 >= len(buffer):
                break
            escape = buffer[i + 1]
            if escape != "u":
                out.append(JSON_ESCAPES.get(escape, escape))
                i += 2
                continue
            code, consumed = unicode_escape(buffer, i)
            if consumed == 0:
                break
            out.append(chr(code))
            i += consumed
        return "".join(out), i


def unicode_escape(buffer: str, start: int) -> tuple[int, int]:
    high = hex_code(buffer, start)
    if high is None:
        return 0, 0
    if not 0xD800 <= high < 0xDC00:
        return high, 6
    if len(buffer) < start + 12:
        return 0, 0
    low = hex_code(buffer, start + 6)
    if low is None or not 0xDC00 <= low < 0xE000:
        return 0xFFFD, 6
    return 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00), 12


def hex_code(buffer: str, start: int) -> int | None:
    digits = buffer[start + 2 : start + 6]
    if buffer[start : start + 2] != "\\u" or len(digits) < 4:
        return None
    try:
        return int(digits, 16)
    except ValueError:
        return 0xFFFD
