from app.services.wer import tokenize, word_errors


def test_wer_counts_substitution_insertion_deletion() -> None:
    result = word_errors("какой лимит размера файла", "какой лимит у файла сейчас")

    assert (result.errors, result.words) == (2, 4)
    assert result.wer == 0.5


def test_wer_ignores_case_punctuation_yo_and_number_format() -> None:
    assert word_errors("Ещё 20 дней!", "еще двадцать дней").errors == 0
    assert tokenize("Отпуск — 28 дней.") == ["отпуск", "двадцать", "восемь", "дней"]


def test_empty_reference() -> None:
    assert word_errors("", "").wer == 0.0
