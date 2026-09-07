import pytest

from app.parsing.markdown import parse_markdown
from app.parsing.models import DocumentParseError

DOCUMENT = """---
title: служебные метаданные
---
# Руководство

Вступление в одну
строку, разбитую переносом.

## Установка

Шаги установки:

- установить uv
- выполнить uv sync

```bash
# это комментарий в коде, а не заголовок
uv run pytest
```

Раздел второго уровня
---------------------

Текст подраздела.

### Детали

Глубокий раздел.

# Приложение

Последний абзац.
"""


def test_blocks_and_section_paths() -> None:
    blocks = parse_markdown(DOCUMENT.encode()).blocks

    assert [(b.kind, b.section) for b in blocks] == [
        ("heading", ("Руководство",)),
        ("paragraph", ("Руководство",)),
        ("heading", ("Руководство", "Установка")),
        ("paragraph", ("Руководство", "Установка")),
        ("paragraph", ("Руководство", "Установка")),
        ("code", ("Руководство", "Установка")),
        ("heading", ("Руководство", "Раздел второго уровня")),
        ("paragraph", ("Руководство", "Раздел второго уровня")),
        ("heading", ("Руководство", "Раздел второго уровня", "Детали")),
        ("paragraph", ("Руководство", "Раздел второго уровня", "Детали")),
        ("heading", ("Приложение",)),
        ("paragraph", ("Приложение",)),
    ]


def test_paragraph_lines_are_joined_and_lists_keep_lines() -> None:
    paragraphs = [b.text for b in parse_markdown(DOCUMENT.encode()).blocks if b.kind == "paragraph"]

    assert paragraphs[0] == "Вступление в одну строку, разбитую переносом."
    assert paragraphs[2] == "- установить uv\n- выполнить uv sync"


def test_code_block_is_kept_verbatim() -> None:
    code = next(b for b in parse_markdown(DOCUMENT.encode()).blocks if b.kind == "code")

    assert code.text == "# это комментарий в коде, а не заголовок\nuv run pytest"


def test_front_matter_is_removed() -> None:
    text = "\n".join(b.text for b in parse_markdown(DOCUMENT.encode()).blocks)

    assert "служебные метаданные" not in text


def test_windows_line_endings_and_bom() -> None:
    blocks = parse_markdown("﻿# Заголовок\r\n\r\nТекст\r\n".encode()).blocks

    assert [(b.kind, b.text) for b in blocks] == [("heading", "Заголовок"), ("paragraph", "Текст")]


def test_unclosed_code_fence_is_kept() -> None:
    blocks = parse_markdown(b"```\nprint(1)\n").blocks

    assert [(b.kind, b.text) for b in blocks] == [("code", "print(1)")]


@pytest.mark.parametrize("content", ["", "# Только заголовок\n", "\n\n   \n"])
def test_document_without_text_is_rejected(content: str) -> None:
    with pytest.raises(DocumentParseError) as error:
        parse_markdown(content.encode())

    assert error.value.code == "no_text"
