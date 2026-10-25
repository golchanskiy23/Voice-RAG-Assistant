from pathlib import Path

import pytest

from app.core.config import PROMPTS_DIR
from app.services.context import fit_to_budget, format_context, make_quote
from app.services.llm_answer import NO_ANSWER, LLMAnswer
from app.services.prompts import load_answer_prompt
from tests.factories import count_words
from tests.llm_fakes import PROMPT, hit


def test_system_prompt_enforces_grounding_and_refusal() -> None:
    assert NO_ANSWER in PROMPT.system
    assert "только" in PROMPT.system
    for name in LLMAnswer.model_fields:
        assert f'"{name}"' in PROMPT.system


def test_user_prompt_keeps_dollar_signs_from_question() -> None:
    _, user = PROMPT.messages(question="Сколько стоит $5?", context="[1] (файл: a.md)\nтекст")

    assert user.content.startswith("Контекст:")
    assert "Сколько стоит $5?" in user.content


def test_unknown_prompt_version_fails_fast() -> None:
    with pytest.raises(FileNotFoundError, match="v999"):
        load_answer_prompt(PROMPTS_DIR, "v999")


def test_prompt_version_comes_from_folder(tmp_path: Path) -> None:
    folder = tmp_path / "answer" / "v2"
    folder.mkdir(parents=True)
    for name in ("system", "user", "retry"):
        (folder / f"{name}.md").write_text(f"{name} $question $context $errors", encoding="utf-8")

    assert load_answer_prompt(tmp_path, "v2").version == "answer/v2"


def test_context_numbers_chunks_with_metadata() -> None:
    context = format_context([hit("один", page=None), hit("два", filename="b.pdf", page=3)])

    assert context == (
        "[1] (файл: guide.md, раздел: Раздел)\nодин\n\n"
        "[2] (файл: b.pdf, стр. 3, раздел: Раздел)\nдва"
    )


def test_budget_keeps_rank_order_and_stops_at_first_overflow() -> None:
    hits = [hit("коротко"), hit("очень " * 50), hit("коротко")]

    assert fit_to_budget(hits, budget=20, count_tokens=count_words) == hits[:1]


def test_quote_is_limited_to_200_chars_on_word_boundary() -> None:
    quote = make_quote("слово " * 100)

    assert len(quote) <= 200
    assert quote.endswith("слово…")
    assert make_quote("  короткий\n текст ") == "короткий текст"
