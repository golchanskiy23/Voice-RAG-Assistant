import asyncio
import json
import random
from pathlib import Path
from typing import Any

import pytest

from app.adapters.llm.base import LLMClient
from app.adapters.llm.registry import build_llm
from app.adapters.vector_stores.base import SearchHit
from app.parsing.markdown import parse_markdown
from app.services.chunking import Chunker
from app.services.llm_answer import NO_ANSWER
from app.services.prompts import load_answer_prompt
from app.services.rag import RAGService, RAGSettings
from tests.factories import count_words, make_settings
from tests.llm_fakes import hit, make_search

API_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = API_DIR.parents[1] / "data"
pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not DATA_DIR.is_dir(), reason="нет корпуса data/ под eval/questions.jsonl"),
]
QUESTIONS = [
    json.loads(line)
    for line in (API_DIR / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
]
IN_SCOPE = [q for q in QUESTIONS if q["expected"]][:8]
OUT_OF_SCOPE = [q for q in QUESTIONS if not q["expected"]]


def corpus() -> list[SearchHit]:
    hits = []
    for path in sorted(DATA_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        chunker = Chunker(count_words, 300, 45)
        for draft in chunker.split(parse_markdown(path.read_bytes()).blocks):
            hits.append(hit(draft.text, filename=path.name, page=None))
    return hits


@pytest.fixture(scope="module")
def llm() -> LLMClient:
    settings = make_settings()
    if settings.llm_provider == "stub":
        pytest.skip("нужен реальный LLM: LLM_PROVIDER и ключ в окружении")
    return build_llm(settings)


def answer(llm: LLMClient, question: str, hits: list[SearchHit]) -> tuple[str, list[str]]:
    settings = make_settings()
    prompt = load_answer_prompt(settings.prompts_dir, settings.prompt_version)
    service = RAGService(make_search(hits), llm, prompt, count_words, RAGSettings(top_k=len(hits)))
    result = asyncio.run(service.answer(question))
    return result.text, [s.filename for s in result.sources]


@pytest.mark.parametrize("case", IN_SCOPE, ids=lambda q: q["id"])
def test_answer_cites_document_with_evidence(llm: LLMClient, case: dict[str, Any]) -> None:
    chunks = corpus()
    evidence = case["expected"][0]
    relevant = next(h for h in chunks if evidence["contains"] in " ".join(h.text.split()))
    distractors = random.Random(case["id"]).sample([h for h in chunks if h is not relevant], 3)
    context = [*distractors[:1], relevant, *distractors[1:]]

    text, files = answer(llm, case["question"], context)

    assert text != NO_ANSWER
    assert evidence["file"] in files


@pytest.mark.parametrize("case", OUT_OF_SCOPE, ids=lambda q: q["id"])
def test_refuses_question_outside_documents(llm: LLMClient, case: dict[str, Any]) -> None:
    context = random.Random(case["id"]).sample(corpus(), 4)

    text, files = answer(llm, case["question"], context)

    assert text == NO_ANSWER
    assert files == []
