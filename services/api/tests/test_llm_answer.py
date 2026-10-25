import pytest

from app.services.llm_answer import AnswerValidationError, parse_llm_answer


def test_parses_valid_answer() -> None:
    result = parse_llm_answer(
        '{"answer": "  Ответ  ", "sources": [1, 3], "confidence": 0.8}', chunk_count=3
    )

    assert result.answer == "Ответ"
    assert result.sources == [1, 3]
    assert result.confidence == 0.8


def test_refusal_without_sources_is_valid() -> None:
    result = parse_llm_answer('{"answer": "Нет ответа", "confidence": 0}', chunk_count=0)

    assert result.sources == []


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ('{"answer": "a", "sources": [4], "confidence": 0.5}', "[4] вне контекста"),
        ('{"answer": "a", "sources": [0, 2], "confidence": 0.5}', "[0] вне контекста"),
        ('{"answer": "a", "sources": [1], "confidence": 1.5}', "confidence"),
        ('{"answer": "", "confidence": 0.5}', "answer"),
        ('{"answer": "a", "confidence": 0.5, "extra": 1}', "extra"),
        ('{"answer": "a", "sources": ["x"], "confidence": 0.5}', "sources.0"),
        ("не JSON", "Invalid JSON"),
    ],
)
def test_rejects_invalid_answer(raw: str, fragment: str) -> None:
    with pytest.raises(AnswerValidationError) as exc_info:
        parse_llm_answer(raw, chunk_count=3)

    assert fragment in exc_info.value.details
