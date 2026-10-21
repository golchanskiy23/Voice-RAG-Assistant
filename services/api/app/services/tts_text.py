import re

UNITS_MASCULINE = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
UNITS_FEMININE = ["", "одна", "две", *UNITS_MASCULINE[3:]]
TEENS = [
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
]
TENS = [
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
]
HUNDREDS = [
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
]
SCALES: list[tuple[tuple[str, str, str], bool]] = [
    (("", "", ""), False),
    (("тысяча", "тысячи", "тысяч"), True),
    (("миллион", "миллиона", "миллионов"), False),
    (("миллиард", "миллиарда", "миллиардов"), False),
]
MAX_NUMBER = 10**12 - 1

UNIT_WORDS = {
    "%": "процентов",
    "₽": "рублей",
    "€": "евро",
    "$": "долларов",
    "МБ": "мегабайт",
    "ГБ": "гигабайт",
    "КБ": "килобайт",
    "мс": "миллисекунд",
}

CITATION = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\S+@\S+\.\w+")
MARKDOWN_MARKS = re.compile(r"[*_`#>|~]+")
LIST_MARKER = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.M)
TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
GROUPED_NUMBER = re.compile(r"\b(\d{1,3}(?:[  ]\d{3})+)\b")
DECIMAL = re.compile(r"\b(\d+)[.,](\d+)\b")
NUMBER_WITH_UNIT = re.compile(r"\b(\d+)\s*(%|₽|€|\$|МБ|ГБ|КБ|мс)(?!\w)")
INTEGER = re.compile(r"\b\d+\b")
SPACES = re.compile(r"[ \t]+")


def plural(n: int, forms: tuple[str, str, str]) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def triple_to_words(n: int, feminine: bool) -> list[str]:
    words = [HUNDREDS[n // 100]]
    rest = n % 100
    if 10 <= rest < 20:
        words.append(TEENS[rest - 10])
    else:
        words.append(TENS[rest // 10])
        words.append((UNITS_FEMININE if feminine else UNITS_MASCULINE)[rest % 10])
    return [w for w in words if w]


def number_to_words(n: int, feminine: bool = False) -> str:
    if n == 0:
        return "ноль"
    if n < 0:
        return "минус " + number_to_words(-n, feminine)
    if n > MAX_NUMBER:
        return " ".join(number_to_words(int(d)) for d in str(n))
    words: list[str] = []
    for scale, (forms, scale_feminine) in reversed(list(enumerate(SCALES))):
        part = n // 1000**scale % 1000
        if not part:
            continue
        words.extend(triple_to_words(part, feminine if scale == 0 else scale_feminine))
        if scale:
            words.append(plural(part, forms))
    return " ".join(words)


def decimal_to_words(whole: str, fraction: str) -> str:
    denominators = {1: "десятых", 2: "сотых", 3: "тысячных"}
    if len(fraction) not in denominators:
        return f"{number_to_words(int(whole))} точка {number_to_words(int(fraction))}"
    integer = int(whole)
    whole_word = plural(integer, ("целая", "целых", "целых"))
    fraction_words = number_to_words(int(fraction), feminine=True)
    return (
        f"{number_to_words(integer, feminine=True)} {whole_word} "
        f"{fraction_words} {denominators[len(fraction)]}"
    )


def time_to_words(hours: str, minutes: str) -> str:
    spoken_minutes = (
        "ноль ноль" if minutes == "00" else number_to_words(int(minutes), feminine=True)
    )
    return f"{number_to_words(int(hours))} {spoken_minutes}"


def normalize_for_tts(text: str) -> str:
    text = MARKDOWN_LINK.sub(r"\1", text)
    text = CITATION.sub("", text)
    text = URL.sub("", text)
    text = EMAIL.sub("адрес в тексте ответа", text)
    text = LIST_MARKER.sub("", text)
    text = MARKDOWN_MARKS.sub("", text)
    text = TIME.sub(lambda m: time_to_words(m[1], m[2]), text)
    text = GROUPED_NUMBER.sub(lambda m: re.sub(r"[  ]", "", m[1]), text)
    text = NUMBER_WITH_UNIT.sub(lambda m: f"{number_to_words(int(m[1]))} {UNIT_WORDS[m[2]]}", text)
    text = DECIMAL.sub(lambda m: decimal_to_words(m[1], m[2]), text)
    text = INTEGER.sub(lambda m: number_to_words(int(m[0])), text)
    text = text.replace("—", ",").replace("–", ",")
    lines = [SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line).replace(" ,", ",").replace(" .", ".")
