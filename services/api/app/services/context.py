from collections.abc import Callable, Sequence

from app.adapters.vector_stores.base import SearchHit

QUOTE_LIMIT = 200


def fit_to_budget(
    hits: Sequence[SearchHit],
    budget: int,
    count_tokens: Callable[[str], int],
    render: Callable[[int, SearchHit], str],
) -> list[SearchHit]:
    selected: list[SearchHit] = []
    used = 0
    for hit in hits:
        cost = count_tokens(render(len(selected) + 1, hit))
        if used + cost > budget:
            break
        selected.append(hit)
        used += cost
    return selected


def make_quote(text: str, limit: int = QUOTE_LIMIT) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[: limit - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut) + "…"
