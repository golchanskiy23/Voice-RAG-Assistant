import re

from app.parsing.models import Block, DocumentParseError, ParsedDocument

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
SETEXT_UNDERLINE = re.compile(r"^(=+|-+)\s*$")
FENCE = re.compile(r"^(```|~~~)")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")


def parse_markdown(content: bytes) -> ParsedDocument:
    text = FRONT_MATTER.sub("", content.decode("utf-8").replace("\r\n", "\n").lstrip("﻿"))
    blocks: list[Block] = []
    section: tuple[str, ...] = ()
    buffer: list[str] = []
    fence: str | None = None
    code: list[str] = []

    def flush() -> None:
        paragraph = "\n".join(buffer).strip()
        if paragraph:
            is_list = all(
                LIST_ITEM.match(line) or line.startswith(" ") for line in buffer if line.strip()
            )
            joined = paragraph if is_list else " ".join(paragraph.split())
            blocks.append(Block("paragraph", joined, section))
        buffer.clear()

    def heading(level: int, title: str) -> None:
        nonlocal section
        flush()
        section = (*section[: level - 1], title)
        blocks.append(Block("heading", title, section, level=level))

    lines = text.split("\n")
    for index, line in enumerate(lines):
        if fence is not None:
            if line.startswith(fence):
                blocks.append(Block("code", "\n".join(code), section))
                fence, code = None, []
            else:
                code.append(line)
            continue

        fence_match = FENCE.match(line)
        if fence_match:
            flush()
            fence = fence_match.group(1)
            continue

        atx = ATX_HEADING.match(line)
        if atx and atx.group(2):
            heading(len(atx.group(1)), atx.group(2))
            continue

        underline = SETEXT_UNDERLINE.match(line)
        if underline and len(buffer) == 1 and buffer[0].strip() and index > 0:
            title = buffer.pop().strip()
            heading(1 if underline.group(1).startswith("=") else 2, title)
            continue

        if line.strip():
            buffer.append(line)
        else:
            flush()

    if fence is not None and "".join(code).strip():
        blocks.append(Block("code", "\n".join(code).rstrip("\n"), section))
    flush()

    if not any(block.kind in ("paragraph", "code") for block in blocks):
        raise DocumentParseError("no_text", "В Markdown-файле нет текста")
    return ParsedDocument(blocks=blocks)
