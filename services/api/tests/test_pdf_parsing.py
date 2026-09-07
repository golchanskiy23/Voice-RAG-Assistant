import io
from pathlib import Path

import pikepdf
import pytest

from app.parsing.models import Block, DocumentParseError, ParsedDocument
from app.parsing.pdf import PAGE_NUMBER, join_lines, parse_pdf
from app.parsing.pdf_text import extract_runs
from tests.fixtures.corpus import (
    EN_HEADER,
    EN_SECTIONS,
    EN_TITLE,
    RU_HEADER,
    RU_SECTIONS,
    RU_TITLE,
    paragraphs,
)

FIXTURES = Path(__file__).parent / "fixtures"
RU_PDF = (FIXTURES / "regulation_ru.pdf").read_bytes()
EN_PDF = (FIXTURES / "faq_en_type1.pdf").read_bytes()


def body(document: ParsedDocument) -> list[Block]:
    return [block for block in document.blocks if block.kind == "paragraph"]


@pytest.mark.parametrize(
    ("content", "sections"),
    [(RU_PDF, RU_SECTIONS), (EN_PDF, EN_SECTIONS)],
    ids=["ru-truetype-tounicode", "en-type1-winansi"],
)
def test_text_is_neither_lost_nor_glued(
    content: bytes, sections: list[tuple[str, list[str]]]
) -> None:
    assert [block.text for block in body(parse_pdf(content))] == paragraphs(sections)


def test_hyphenated_words_are_joined() -> None:
    with pikepdf.open(io.BytesIO(RU_PDF)) as pdf:
        raw_lines = [run.text for page in extract_runs(pdf) for run in page]
    text = " ".join(block.text for block in body(parse_pdf(RU_PDF)))

    assert any(line.endswith("-") for line in raw_lines)
    assert "безопасности" in text
    assert "представление" in text


def test_running_headers_and_page_numbers_are_removed() -> None:
    text = "\n".join(block.text for block in parse_pdf(RU_PDF).blocks)

    assert RU_HEADER not in text
    assert "Страница" not in text


def test_paragraph_continues_across_page_break() -> None:
    document = parse_pdf(RU_PDF)

    assert document.pages == 2
    assert [block.pages for block in body(document) if len(block.pages) > 1] == [(1, 2)]


def test_headings_build_section_path() -> None:
    blocks = parse_pdf(RU_PDF).blocks
    headings = [(block.level, block.text) for block in blocks if block.kind == "heading"]
    first_paragraph = body(parse_pdf(RU_PDF))[0]

    assert headings == [(1, RU_TITLE), *((2, title) for title, _ in RU_SECTIONS)]
    assert first_paragraph.section == (RU_TITLE, RU_SECTIONS[0][0])


def test_type1_font_without_tounicode_uses_winansi() -> None:
    document = parse_pdf(EN_PDF)
    text = "\n".join(block.text for block in document.blocks)

    assert EN_HEADER not in text
    assert next(b.text for b in document.blocks if b.kind == "heading") == EN_TITLE


def test_scan_without_text_layer_is_rejected() -> None:
    with pytest.raises(DocumentParseError) as error:
        parse_pdf((FIXTURES / "scan_without_text.pdf").read_bytes())

    assert error.value.code == "no_text"


def test_broken_pdf_is_rejected() -> None:
    with pytest.raises(DocumentParseError) as error:
        parse_pdf(b"%PDF-1.7\nnot really a pdf")

    assert error.value.code == "invalid_pdf"


def test_encrypted_pdf_is_rejected() -> None:
    buffer = io.BytesIO()
    with pikepdf.open(io.BytesIO(RU_PDF)) as pdf:
        pdf.save(buffer, encryption=pikepdf.Encryption(user="secret", owner="secret"))

    with pytest.raises(DocumentParseError) as error:
        parse_pdf(buffer.getvalue())

    assert error.value.code == "encrypted_pdf"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("информацион-", "ной системы", "информационной системы"),
        ("документ", "загружен", "документ загружен"),
        ("Москва -", "столица", "Москва - столица"),
        ("версия 2-", "Б", "версия 2- Б"),
    ],
)
def test_join_lines(left: str, right: str, expected: str) -> None:
    assert join_lines(left, right) == expected


@pytest.mark.parametrize(
    "line", ["3", "- 3 -", "Страница 2 из 10", "стр. 7", "Page 4 of 9", "12/40"]
)
def test_page_number_lines(line: str) -> None:
    assert PAGE_NUMBER.match(line)


@pytest.mark.parametrize("line", ["2. Загрузка документов", "В 2024 году", "Глава 3"])
def test_not_page_numbers(line: str) -> None:
    assert not PAGE_NUMBER.match(line)
