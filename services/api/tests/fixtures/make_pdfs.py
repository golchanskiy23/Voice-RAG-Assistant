from pathlib import Path

from reportlab.lib.pagesizes import A5
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from tests.fixtures.corpus import (
    EN_HEADER,
    EN_SECTIONS,
    EN_TITLE,
    HYPHEN_POINT,
    RU_HEADER,
    RU_SECTIONS,
    RU_TITLE,
)

FIXTURES = Path(__file__).parent
DEJAVU = Path("/usr/share/fonts/TTF/DejaVuSans.ttf")
MARGIN = 40
BODY, HEADING, TITLE = 10.0, 12.5, 15.0
LEADING = 14.0
TOP = A5[1] - 55
BOTTOM = 50

Line = tuple[str, float, float]


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(candidate.replace(HYPHEN_POINT, ""), font, size) <= width:
            current = candidate
            continue
        if HYPHEN_POINT in word:
            head, tail = word.split(HYPHEN_POINT, 1)
            with_head = f"{current} {head}-".strip()
            if pdfmetrics.stringWidth(with_head, font, size) <= width:
                lines.append(with_head)
                current = tail
                continue
        lines.append(current.replace(HYPHEN_POINT, ""))
        current = word
    lines.append(current.replace(HYPHEN_POINT, ""))
    return lines


def layout(
    title: str, sections: list[tuple[str, list[str]]], font: str, bottom: float
) -> list[list[Line]]:
    width = A5[0] - 2 * MARGIN
    items: list[tuple[str, float, float]] = [(title, TITLE, 1.6)]
    for heading, paragraphs in sections:
        items.append((heading, HEADING, 1.4))
        for paragraph in paragraphs:
            for i, line in enumerate(wrap(paragraph, font, BODY, width)):
                items.append((line, BODY, 1.8 if i == 0 else 1.0))
    pages: list[list[Line]] = [[]]
    y = TOP
    for text, size, gap in items:
        y -= LEADING * gap
        if y < bottom:
            pages.append([])
            y = TOP - LEADING
        pages[-1].append((text, size, y))
    return pages


def render(
    path: Path,
    title: str,
    sections: list[tuple[str, list[str]]],
    font: str,
    header: str,
    footer: str,
    bottom: float = BOTTOM,
) -> None:
    pages = layout(title, sections, font, bottom)
    canvas = Canvas(str(path), pagesize=A5, invariant=True)
    for number, lines in enumerate(pages, start=1):
        canvas.setFont(font, 8)
        canvas.drawString(MARGIN, A5[1] - 30, header)
        canvas.drawCentredString(A5[0] / 2, 25, footer.format(page=number, total=len(pages)))
        for text, size, y in lines:
            canvas.setFont(font, size)
            canvas.drawString(MARGIN, y, text)
        canvas.showPage()
    canvas.save()


def render_scan(path: Path) -> None:
    canvas = Canvas(str(path), pagesize=A5, invariant=True)
    canvas.rect(MARGIN, MARGIN, 200, 300, fill=1)
    canvas.showPage()
    canvas.save()


def main() -> None:
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(DEJAVU)))
    render(
        FIXTURES / "regulation_ru.pdf",
        RU_TITLE,
        RU_SECTIONS,
        "DejaVuSans",
        RU_HEADER,
        "Страница {page} из {total}",
        bottom=170,
    )
    render(
        FIXTURES / "faq_en_type1.pdf",
        EN_TITLE,
        EN_SECTIONS,
        "Helvetica",
        EN_HEADER,
        "- {page} -",
        bottom=330,
    )
    render_scan(FIXTURES / "scan_without_text.pdf")


if __name__ == "__main__":
    main()
