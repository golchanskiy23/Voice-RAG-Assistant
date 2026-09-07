import itertools

import pytest

from app.adapters.embedders.hashing import HashingEmbedder
from app.parsing.models import Block
from app.parsing.pdf import parse_pdf
from app.services.chunking import Chunker, split_sentences
from tests.fixtures.corpus import RU_SECTIONS
from tests.test_pdf_parsing import RU_PDF

count = HashingEmbedder().count_tokens


def sentence(n: int, words: int = 8) -> str:
    return " ".join(f"Слово{n}_{i}" if i == 0 else f"слово{n}_{i}" for i in range(words)) + "."


def section_blocks(section: tuple[str, ...], paragraphs: list[str], page: int = 1) -> list[Block]:
    return [
        Block("heading", section[-1], section, (page,), len(section)),
        *(Block("paragraph", p, section, (page,)) for p in paragraphs),
    ]


def test_split_sentences_handles_abbreviations() -> None:
    text = "Срок — 5 дней, т.е. неделя. См. приложение. Готово! Version 2.0 is out."

    assert split_sentences(text) == [
        "Срок — 5 дней, т.е. неделя.",
        "См. приложение.",
        "Готово!",
        "Version 2.0 is out.",
    ]


def test_chunks_respect_size_and_keep_every_sentence() -> None:
    sentences = [sentence(i) for i in range(30)]
    blocks = section_blocks(("Раздел",), [" ".join(sentences[i : i + 5]) for i in range(0, 30, 5)])

    chunks = Chunker(count, chunk_size=40, overlap=8).split(blocks)

    assert all(c.tokens <= 40 for c in chunks)
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)


def test_consecutive_chunks_overlap_by_whole_sentences() -> None:
    blocks = section_blocks(("Раздел",), [" ".join(sentence(i) for i in range(12))])

    chunks = Chunker(count, chunk_size=40, overlap=10).split(blocks)

    for previous, current in itertools.pairwise(chunks):
        last_sentence = split_sentences(previous.text)[-1]
        assert current.text.startswith(last_sentence)
        assert count(last_sentence) <= 10


def test_no_overlap_when_disabled() -> None:
    blocks = section_blocks(("Раздел",), [" ".join(sentence(i) for i in range(12))])

    chunks = Chunker(count, chunk_size=40, overlap=0).split(blocks)
    sentences = [s for c in chunks for s in split_sentences(c.text)]

    assert len(sentences) == len(set(sentences)) == 12


def test_chunks_do_not_cross_sections() -> None:
    blocks = [
        *section_blocks(("Документ", "Первый"), [sentence(1), sentence(2)]),
        *section_blocks(("Документ", "Второй"), [sentence(3)]),
    ]

    chunks = Chunker(count, chunk_size=200, overlap=20).split(blocks)

    assert [(c.section, c.text.count(".")) for c in chunks] == [
        ("Документ > Первый", 2),
        ("Документ > Второй", 1),
    ]
    assert "\n\n" in chunks[0].text


def test_oversized_sentence_is_split_into_windows() -> None:
    long_sentence = sentence(0, words=95)

    chunks = Chunker(count, chunk_size=40, overlap=5).split(
        section_blocks(("Раздел",), [long_sentence])
    )

    assert len(chunks) >= 3
    assert all(c.tokens <= 40 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == long_sentence.split()


def test_code_block_is_not_split_into_sentences() -> None:
    code = "x = 1. y = 2. print(x + y)"
    blocks = [Block("code", code, ("Пример",))]

    assert [c.text for c in Chunker(count, chunk_size=50, overlap=5).split(blocks)] == [code]


def test_pages_follow_source_blocks() -> None:
    blocks = [
        Block("paragraph", sentence(1), ("Раздел",), (1,)),
        Block("paragraph", sentence(2), ("Раздел",), (1, 2)),
        Block("paragraph", sentence(3), ("Раздел",), (3,)),
    ]

    chunks = Chunker(count, chunk_size=20, overlap=0).split(blocks)

    assert [(c.page, c.pages) for c in chunks] == [(1, (1, 2)), (3, (3,))]


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, 150), (100, -1)])
def test_invalid_overlap_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="перекрытие"):
        Chunker(count, chunk_size=size, overlap=overlap)


def test_real_pdf_chunks_carry_source_metadata() -> None:
    chunks = Chunker(count, chunk_size=60, overlap=10).split(parse_pdf(RU_PDF).blocks)

    sections = {c.section for c in chunks}
    assert sections == {f"Регламент обработки документов > {title}" for title, _ in RU_SECTIONS}
    assert all(c.page in (1, 2) for c in chunks)
    assert any(c.pages == (1, 2) for c in chunks)
