import io
import itertools
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, replace

import pikepdf

from app.parsing.models import Block, DocumentParseError, ParsedDocument
from app.parsing.pdf_text import TextRun, extract_runs

SAME_LINE_TOLERANCE = 0.5
WORD_GAP = 0.2
PARAGRAPH_GAP = 1.6
HEADING_RATIO = 1.2
HEADING_MAX_CHARS = 150
EDGE_LINES = 2
REPEAT_SHARE = 0.6
PAGE_NUMBER = re.compile(
    r"^\s*(?:(?:стр(?:аница)?|page|с)\.?\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?"
    r"(?:\s*(?:из|of|/)\s*\d{1,4})?\s*$",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"[.!?…:;»\")]\s*$")
LIST_MARKER = re.compile(r"^\s*(?:[-•–—*]|\d{1,3}[.)])\s+")


@dataclass(frozen=True)
class Line:
    page: int
    y: float
    x: float
    size: float
    text: str


def build_lines(runs: list[TextRun]) -> list[Line]:
    lines: list[list[TextRun]] = []
    for run in sorted(runs, key=lambda r: (-r.y, r.x)):
        if lines:
            current = lines[-1]
            if abs(current[0].y - run.y) <= SAME_LINE_TOLERANCE * max(run.size, current[0].size):
                current.append(run)
                continue
        lines.append([run])

    result: list[Line] = []
    for group in lines:
        group.sort(key=lambda r: r.x)
        text = group[0].text
        for prev, run in itertools.pairwise(group):
            gap = run.x - prev.x_end
            needs_space = (
                gap > WORD_GAP * run.size
                and not text.endswith(" ")
                and not run.text.startswith(" ")
            )
            text += (" " if needs_space else "") + run.text
        size = max(group, key=lambda r: len(r.text)).size
        result.append(
            Line(
                page=group[0].page,
                y=group[0].y,
                x=group[0].x,
                size=size,
                text=" ".join(text.split()),
            )
        )
    return [line for line in result if line.text]


def normalize_repeat(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def remove_page_furniture(pages: list[list[Line]]) -> list[list[Line]]:
    edges = [page[:EDGE_LINES] + page[-EDGE_LINES:] for page in pages]
    counts = Counter(
        normalize_repeat(line.text) for edge in edges for line in {id(x): x for x in edge}.values()
    )
    threshold = max(2, math.ceil(REPEAT_SHARE * len(pages)))
    repeated = {text for text, count in counts.items() if count >= threshold}

    cleaned: list[list[Line]] = []
    for page, edge in zip(pages, edges, strict=True):
        edge_ids = {id(line) for line in edge}
        cleaned.append(
            [
                line
                for line in page
                if id(line) not in edge_ids
                or not (PAGE_NUMBER.match(line.text) or normalize_repeat(line.text) in repeated)
            ]
        )
    return cleaned


def join_lines(left: str, right: str) -> str:
    if re.search(r"\w[-­]$", left) and right[:1].islower():
        return left[:-1] + right
    return f"{left} {right}"


def body_size(lines: list[Line]) -> float:
    weights: Counter[float] = Counter()
    for line in lines:
        weights[round(line.size, 1)] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def is_heading(line: Line, body: float) -> bool:
    return (
        body > 0
        and line.size >= body * HEADING_RATIO
        and len(line.text) <= HEADING_MAX_CHARS
        and not line.text.endswith((".", ",", ";"))
    )


def typical_leading(pages: list[list[Line]]) -> float:
    gaps = [
        prev.y - line.y
        for page in pages
        for prev, line in itertools.pairwise(page)
        if 0 < prev.y - line.y < 3 * max(prev.size, line.size)
    ]
    return statistics.median(gaps) if gaps else 0.0


def build_blocks(pages: list[list[Line]]) -> list[Block]:
    all_lines = [line for page in pages for line in page]
    body = body_size(all_lines)
    leading = typical_leading(pages)
    heading_sizes = sorted(
        {round(line.size, 1) for line in all_lines if is_heading(line, body)}, reverse=True
    )

    blocks: list[Block] = []
    section: tuple[str, ...] = ()
    paragraph: list[str] = []
    paragraph_pages: list[int] = []
    previous: Line | None = None

    def flush() -> None:
        if paragraph:
            text = paragraph[0]
            for line_text in paragraph[1:]:
                text = join_lines(text, line_text)
            blocks.append(Block("paragraph", text, section, tuple(dict.fromkeys(paragraph_pages))))
            paragraph.clear()
            paragraph_pages.clear()

    for line in all_lines:
        if is_heading(line, body):
            flush()
            level = heading_sizes.index(round(line.size, 1)) + 1
            if (
                blocks
                and blocks[-1].kind == "heading"
                and blocks[-1].pages == (line.page,)
                and blocks[-1].level == level
            ):
                merged = f"{blocks[-1].text} {line.text}"
                section = (*section[:-1], merged)
                blocks[-1] = replace(blocks[-1], text=merged, section=section)
            else:
                section = (*section[: level - 1], line.text)
                blocks.append(Block("heading", line.text, section, (line.page,), level))
            previous = line
            continue

        starts_new = previous is None or not paragraph or LIST_MARKER.match(line.text) is not None
        if previous is not None and paragraph and not starts_new:
            if line.page == previous.page:
                starts_new = leading > 0 and previous.y - line.y > PARAGRAPH_GAP * leading
            else:
                starts_new = bool(SENTENCE_END.search(paragraph[-1])) or not line.text[:1].islower()
        if starts_new:
            flush()
        paragraph.append(line.text)
        paragraph_pages.append(line.page)
        previous = line
    flush()
    return blocks


def parse_pdf(content: bytes) -> ParsedDocument:
    try:
        pdf = pikepdf.open(io.BytesIO(content))
    except pikepdf.PasswordError:
        raise DocumentParseError("encrypted_pdf", "PDF защищён паролем") from None
    except pikepdf.PdfError:
        raise DocumentParseError("invalid_pdf", "Не удалось прочитать PDF") from None
    with pdf:
        pages = [build_lines(runs) for runs in extract_runs(pdf)]
        page_count = len(pdf.pages)
    blocks = build_blocks(remove_page_furniture(pages))
    if not any(block.kind == "paragraph" for block in blocks):
        raise DocumentParseError(
            "no_text", "В PDF нет текстового слоя (скан без OCR не поддерживается)"
        )
    return ParsedDocument(blocks=blocks, pages=page_count)
