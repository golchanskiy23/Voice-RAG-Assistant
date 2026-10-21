import re

SENTENCE_END = re.compile(r"(?<!\d)(?<!\b[а-яa-z])([.!?…]+)(?=\s+[\"«(]?[А-ЯЁA-Z0-9])")
SOFT_BREAK = re.compile(r"[,;:—]\s")


class SentenceBuffer:
    def __init__(
        self, min_chars: int = 20, max_chars: int = 220, first_max_chars: int | None = None
    ) -> None:
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._first_max_chars = first_max_chars
        self._emitted = 0
        self._text = ""

    def feed(self, delta: str) -> list[str]:
        self._text += delta
        ready: list[str] = []
        while sentence := self._next():
            ready.append(sentence)
        return ready

    def flush(self) -> list[str]:
        rest, self._text = self._text.strip(), ""
        return [rest] if rest else []

    def _next(self) -> str | None:
        first = self._emitted == 0 and bool(self._first_max_chars)
        limit = self._first_max_chars if first and self._first_max_chars else self._max_chars
        ends = [m.end() for m in SENTENCE_END.finditer(self._text) if m.end() >= self._min_chars]
        if first:
            ends += [
                m.end()
                for m in SOFT_BREAK.finditer(self._text, 0, limit)
                if m.end() >= self._min_chars
            ]
        if ends:
            return self._take(min(ends))
        if len(self._text) > limit:
            breaks = [m.end() for m in SOFT_BREAK.finditer(self._text, 0, limit)]
            cut = breaks[-1] if breaks else self._text.rfind(" ", 0, limit)
            if cut > self._min_chars:
                return self._take(cut)
        return None

    def _take(self, end: int) -> str:
        sentence, self._text = self._text[:end].strip(), self._text[end:].lstrip()
        self._emitted += 1
        return sentence
