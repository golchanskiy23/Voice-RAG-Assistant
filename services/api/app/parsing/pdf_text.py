import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pikepdf
from pikepdf import Array, Dictionary, Name, Object, Stream, String

Matrix = tuple[float, float, float, float, float, float]

IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
SPACE_ADJUSTMENT = 250.0
MAX_FORM_DEPTH = 8
SIMPLE_ENCODINGS = {
    "/WinAnsiEncoding": "cp1252",
    "/StandardEncoding": "cp1252",
    "/MacRomanEncoding": "mac_roman",
}
HEX = re.compile(r"<([0-9A-Fa-f\s]*)>")
BFRANGE = re.compile(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[.*?\]|<[0-9A-Fa-f\s]*>)", re.S)


def multiply(m: Matrix, n: Matrix) -> Matrix:
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (
        a * a2 + b * c2,
        a * b2 + b * d2,
        c * a2 + d * c2,
        c * b2 + d * d2,
        e * a2 + f * c2 + e2,
        e * b2 + f * d2 + f2,
    )


def as_matrix(values: Any) -> Matrix:
    a, b, c, d, e, f = (float(x) for x in values)
    return (a, b, c, d, e, f)


def translate(tx: float, ty: float) -> Matrix:
    return (1.0, 0.0, 0.0, 1.0, tx, ty)


def scale_of(m: Matrix) -> float:
    return float(abs(m[0] * m[3] - m[1] * m[2]) ** 0.5)


def hex_bytes(value: str) -> bytes:
    return bytes.fromhex(re.sub(r"\s", "", value))


def utf16(value: str) -> str:
    raw = hex_bytes(value)
    return raw.decode("utf-16-be", "replace") if raw else ""


def parse_tounicode(cmap: Stream) -> tuple[dict[int, str], int]:
    data = cmap.read_bytes().decode("latin-1")
    width = 1
    for block in re.findall(r"begincodespacerange(.*?)endcodespacerange", data, re.S):
        for value in HEX.findall(block):
            width = max(width, len(re.sub(r"\s", "", value)) // 2)

    table: dict[int, str] = {}
    for block in re.findall(r"beginbfchar(.*?)endbfchar", data, re.S):
        values = HEX.findall(block)
        for src, dst in zip(values[0::2], values[1::2], strict=False):
            table[int(re.sub(r"\s", "", src), 16)] = utf16(dst)

    for block in re.findall(r"beginbfrange(.*?)endbfrange", data, re.S):
        for lo_hex, hi_hex, dst in BFRANGE.findall(block):
            lo, hi = int(lo_hex, 16), int(hi_hex, 16)
            if dst.startswith("["):
                for offset, value in enumerate(HEX.findall(dst)):
                    table[lo + offset] = utf16(value)
                continue
            base = utf16(dst[1:-1])
            for code in range(lo, hi + 1):
                table[code] = base[:-1] + chr(ord(base[-1]) + code - lo) if base else ""
    return table, width


def parse_cid_widths(widths: Array) -> dict[int, float]:
    result: dict[int, float] = {}
    items = list(widths)
    i = 0
    while i < len(items):
        first = int(items[i])
        nxt = items[i + 1]
        if isinstance(nxt, Array):
            for offset, value in enumerate(nxt):
                result[first + offset] = float(value)
            i += 2
        else:
            last, width = int(nxt), float(items[i + 2])
            for code in range(first, last + 1):
                result[code] = width
            i += 3
    return result


@dataclass
class FontInfo:
    to_unicode: dict[int, str] = field(default_factory=dict)
    code_width: int = 1
    widths: dict[int, float] = field(default_factory=dict)
    default_width: float = 500.0
    encoding: str | None = None

    @classmethod
    def from_pdf(cls, font: Any) -> "FontInfo":
        if not isinstance(font, Dictionary):
            return cls()
        is_composite = font.get("/Subtype") == Name.Type0
        info = cls(code_width=2 if is_composite else 1)

        cmap = font.get("/ToUnicode")
        if isinstance(cmap, Stream):
            info.to_unicode, width = parse_tounicode(cmap)
            info.code_width = max(info.code_width, width) if is_composite else 1

        if is_composite:
            descendant = font.get("/DescendantFonts")
            cid_font = descendant[0] if isinstance(descendant, Array) and len(descendant) else None
            if isinstance(cid_font, Dictionary):
                info.default_width = float(cid_font.get("/DW", 1000))
                widths = cid_font.get("/W")
                if isinstance(widths, Array):
                    info.widths = parse_cid_widths(widths)
            return info

        first = int(font.get("/FirstChar", 0))
        widths = font.get("/Widths")
        if isinstance(widths, Array):
            info.widths = {first + i: float(w) for i, w in enumerate(widths)}
        encoding = font.get("/Encoding")
        if isinstance(encoding, Dictionary):
            encoding = encoding.get("/BaseEncoding")
        info.encoding = SIMPLE_ENCODINGS.get(str(encoding), "cp1252")
        return info

    def codes(self, raw: bytes) -> Iterator[int]:
        step = self.code_width
        for i in range(0, len(raw) - step + 1, step):
            yield int.from_bytes(raw[i : i + step], "big")

    def text(self, code: int) -> str:
        if code in self.to_unicode:
            return self.to_unicode[code]
        if self.encoding and self.code_width == 1:
            return bytes([code]).decode(self.encoding, "replace")
        return ""

    def width(self, code: int) -> float:
        return self.widths.get(code, self.default_width)


@dataclass(frozen=True)
class TextRun:
    page: int
    x: float
    y: float
    x_end: float
    size: float
    text: str


@dataclass
class TextState:
    font: FontInfo = field(default_factory=FontInfo)
    font_size: float = 12.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scale: float = 1.0
    leading: float = 0.0
    rise: float = 0.0


@dataclass
class GraphicsState:
    ctm: Matrix = IDENTITY
    text: TextState = field(default_factory=TextState)

    def copy(self) -> "GraphicsState":
        return GraphicsState(ctm=self.ctm, text=TextState(**vars(self.text)))


class PageTextExtractor:
    def __init__(self, page_number: int) -> None:
        self.page_number = page_number
        self.runs: list[TextRun] = []
        self._fonts: dict[tuple[int, int] | int, FontInfo] = {}

    def extract(
        self, content: Object, resources: Any, ctm: Matrix = IDENTITY, depth: int = 0
    ) -> None:
        state = GraphicsState(ctm=ctm)
        stack: list[GraphicsState] = []
        tm: Matrix = IDENTITY
        tlm: Matrix = IDENTITY
        fonts = resources.get("/Font") if isinstance(resources, Dictionary) else None
        xobjects = resources.get("/XObject") if isinstance(resources, Dictionary) else None

        for instruction in pikepdf.parse_content_stream(content):
            if isinstance(instruction, pikepdf.ContentStreamInlineImage):
                continue
            operands, op = list(instruction.operands), str(instruction.operator)
            if op == "q":
                stack.append(state.copy())
            elif op == "Q" and stack:
                state = stack.pop()
            elif op == "cm" and len(operands) == 6:
                state.ctm = multiply(as_matrix(operands), state.ctm)
            elif op == "BT":
                tm = tlm = IDENTITY
            elif op == "Tf" and len(operands) >= 2:
                state.text.font = self._font(fonts, operands[0])
                state.text.font_size = float(operands[1])
            elif op == "Tc" and operands:
                state.text.char_spacing = float(operands[0])
            elif op == "Tw" and operands:
                state.text.word_spacing = float(operands[0])
            elif op == "Tz" and operands:
                state.text.horizontal_scale = float(operands[0]) / 100
            elif op == "TL" and operands:
                state.text.leading = float(operands[0])
            elif op == "Ts" and operands:
                state.text.rise = float(operands[0])
            elif op == "Tm" and len(operands) == 6:
                tm = tlm = as_matrix(operands)
            elif op in ("Td", "TD") and len(operands) >= 2:
                tx, ty = float(operands[0]), float(operands[1])
                if op == "TD":
                    state.text.leading = -ty
                tm = tlm = multiply(translate(tx, ty), tlm)
            elif op == "T*":
                tm = tlm = multiply(translate(0, -state.text.leading), tlm)
            elif op in ("Tj", "TJ", "'", '"'):
                if op in ("'", '"'):
                    if op == '"' and len(operands) == 3:
                        state.text.word_spacing = float(operands[0])
                        state.text.char_spacing = float(operands[1])
                    tm = tlm = multiply(translate(0, -state.text.leading), tlm)
                tm = self._show(operands, op, state, tm)
            elif op == "Do" and operands and depth < MAX_FORM_DEPTH:
                self._form(xobjects, operands[0], resources, state.ctm, depth)

    def _font(self, fonts: Any, name: Object) -> FontInfo:
        font = fonts.get(str(name)) if isinstance(fonts, Dictionary) else None
        if font is None:
            return FontInfo()
        key = font.objgen if font.is_indirect else id(font)
        if key not in self._fonts:
            self._fonts[key] = FontInfo.from_pdf(font)
        return self._fonts[key]

    def _form(self, xobjects: Any, name: Object, resources: Any, ctm: Matrix, depth: int) -> None:
        xobject = xobjects.get(str(name)) if isinstance(xobjects, Dictionary) else None
        if not isinstance(xobject, Stream) or xobject.get("/Subtype") != Name.Form:
            return
        matrix = xobject.get("/Matrix")
        form_matrix = as_matrix(matrix) if isinstance(matrix, Array) else IDENTITY
        self.extract(
            xobject, xobject.get("/Resources", resources), multiply(form_matrix, ctm), depth + 1
        )

    def _show(self, operands: list[Object], op: str, state: GraphicsState, tm: Matrix) -> Matrix:
        text_state = state.text
        items: list[Object] = (
            list(operands[0]) if op == "TJ" and isinstance(operands[0], Array) else [operands[-1]]
        )
        start = multiply(tm, state.ctm)
        size = text_state.font_size * scale_of(multiply(tm, state.ctm))
        parts: list[str] = []
        for item in items:
            if isinstance(item, String):
                for code in text_state.font.codes(bytes(item)):
                    parts.append(text_state.font.text(code))
                    advance = (
                        text_state.font.width(code) / 1000 * text_state.font_size
                        + text_state.char_spacing
                    )
                    if code == 32 and text_state.font.code_width == 1:
                        advance += text_state.word_spacing
                    tm = multiply(translate(advance * text_state.horizontal_scale, 0), tm)
            else:
                adjustment = float(item)
                if adjustment < -SPACE_ADJUSTMENT and parts and not parts[-1].endswith(" "):
                    parts.append(" ")
                shift = -adjustment / 1000 * text_state.font_size * text_state.horizontal_scale
                tm = multiply(translate(shift, 0), tm)
        text = "".join(parts)
        if text.strip():
            end = multiply(tm, state.ctm)
            self.runs.append(
                TextRun(
                    page=self.page_number,
                    x=start[4],
                    y=start[5],
                    x_end=end[4],
                    size=size,
                    text=text,
                )
            )
        return tm


def extract_runs(pdf: pikepdf.Pdf) -> Iterator[list[TextRun]]:
    for number, page in enumerate(pdf.pages, start=1):
        extractor = PageTextExtractor(number)
        if page.obj.get("/Contents") is not None:
            extractor.extract(page.obj, page.obj.get("/Resources"))
        yield extractor.runs
