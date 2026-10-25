from collections.abc import Callable, Sequence

from app.adapters.vector_stores.base import SearchHit

QUOTE_LIMIT = 200


def chunk_header(number: int, hit: SearchHit) -> str:
    parts = [f"файл: {hit.filename}"]
    if hit.page is not None:
        parts.append(f"стр. {hit.page}")
    if hit.section:
        parts.append(f"раздел: {hit.section}")
    return f"[{number}] ({', '.join(parts)})"


def format_chunk(number: int, hit: SearchHit) -> str:
    return f"{chunk_header(number, hit)}\n{hit.text.strip()}"


def fit_to_budget(
    hits: Sequence[SearchHit], budget: int, count_tokens: Callable[[str], int]
) -> list[SearchHit]:
    selected: list[SearchHit] = []
    used = 0
    for hit in hits:
        cost = count_tokens(format_chunk(len(selected) + 1, hit))
        if used + cost > budget:
            break
        selected.append(hit)
        used += cost
    return selected


def format_context(hits: Sequence[SearchHit]) -> str:
    return "\n\n".join(format_chunk(number, hit) for number, hit in enumerate(hits, start=1))


def make_quote(text: str, limit: int = QUOTE_LIMIT) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[: limit - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut) + "…"
