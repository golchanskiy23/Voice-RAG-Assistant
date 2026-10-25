import re
from dataclasses import dataclass

from app.services.tts_text import normalize_for_tts

WORD = re.compile(r"[\wё]+", re.I)


def tokenize(text: str) -> list[str]:
    return WORD.findall(normalize_for_tts(text).casefold().replace("ё", "е"))


@dataclass(frozen=True)
class WerResult:
    errors: int
    words: int

    @property
    def wer(self) -> float:
        return self.errors / self.words if self.words else 0.0


def word_errors(reference: str, hypothesis: str) -> WerResult:
    ref, hyp = tokenize(reference), tokenize(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return WerResult(errors=previous[-1], words=len(ref))
