import pytest

from app.services.sentences import SentenceBuffer
from app.services.tts_text import normalize_for_tts, number_to_words


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "ноль"),
        (1, "один"),
        (12, "двенадцать"),
        (21, "двадцать один"),
        (100, "сто"),
        (1000, "одна тысяча"),
        (1500, "одна тысяча пятьсот"),
        (2002, "две тысячи два"),
        (11000, "одиннадцать тысяч"),
        (60000, "шестьдесят тысяч"),
        (1_000_000, "один миллион"),
        (2_345_678, "два миллиона триста сорок пять тысяч шестьсот семьдесят восемь"),
    ],
)
def test_number_to_words(n: int, words: str) -> None:
    assert number_to_words(n) == words


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Лимит — 20 МБ [1].", "Лимит, двадцать мегабайт."),
        ("Суточные: 1 500 рублей [1, 2].", "Суточные: одна тысяча пятьсот рублей."),
        ("**Важно**: см. [регламент](http://x.ru).", "Важно: см. регламент."),
        ("С 11:00 до 16:00.", "С одиннадцать ноль ноль до шестнадцать ноль ноль."),
        ("Рост на 15%.", "Рост на пятнадцать процентов."),
        ("Ставка 1,5 раза.", "Ставка одна целая пять десятых раза."),
        ("- первый\n- второй", "первый второй"),
        ("Пишите на support@example.com.", "Пишите на адрес в тексте ответа."),
    ],
)
def test_normalize_for_tts(text: str, spoken: str) -> None:
    assert normalize_for_tts(text) == spoken


def test_sentence_buffer_emits_complete_sentences_as_tokens_arrive() -> None:
    buffer = SentenceBuffer(min_chars=10)
    text = "Отпуск длится 28 дней. Заявление подаётся заранее! Остальное"

    ready = [s for i in range(0, len(text), 3) for s in buffer.feed(text[i : i + 3])]

    assert ready == ["Отпуск длится 28 дней.", "Заявление подаётся заранее!"]
    assert buffer.flush() == ["Остальное"]
    assert buffer.flush() == []


def test_sentence_buffer_keeps_abbreviations_and_decimals_together() -> None:
    buffer = SentenceBuffer(min_chars=5)

    assert buffer.feed("Ставка 1.5 раза, т. е. больше. Далее") == ["Ставка 1.5 раза, т. е. больше."]


def test_sentence_buffer_merges_too_short_sentences() -> None:
    buffer = SentenceBuffer(min_chars=20)

    assert buffer.feed("Да. Это так, и вот почему. Далее") == ["Да. Это так, и вот почему."]


def test_long_sentence_is_cut_at_soft_break() -> None:
    buffer = SentenceBuffer(min_chars=10, max_chars=40)

    ready = buffer.feed(
        "Очень длинное перечисление условий, которые идут подряд без точки и дальше"
    )

    assert ready == ["Очень длинное перечисление условий,"]


def test_first_chunk_is_cut_early_at_comma() -> None:
    buffer = SentenceBuffer(min_chars=10, first_max_chars=60)

    ready = buffer.feed("Отпуск длится двадцать восемь дней, а при ненормированном дне больше. Ещё")

    assert ready == ["Отпуск длится двадцать восемь дней,", "а при ненормированном дне больше."]


def test_first_chunk_prefers_earlier_sentence_end() -> None:
    buffer = SentenceBuffer(min_chars=5, first_max_chars=60)

    assert buffer.feed("Да, это так. Потом, если нужно, ещё. А") == [
        "Да, это так.",
        "Потом, если нужно, ещё.",
    ]
