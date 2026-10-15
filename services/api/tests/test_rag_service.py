import asyncio

import structlog

from app.services.llm_answer import NO_ANSWER
from app.services.rag import (
    FinalEvent,
    RAGService,
    RAGSettings,
    RetrievedEvent,
    StreamEvent,
    TokenEvent,
)
from tests.llm_fakes import PROMPT, ScriptedLLM, hit, make_service

HITS = [
    hit("Файлы больше двадцати мегабайт отклоняются.", filename="faq.md", page=2),
    hit("Ответ озвучивается по предложениям.", filename="guide.pdf", page=7),
]
VALID = '{"answer": "Лимит — 20 МБ [1].", "sources": [1], "confidence": 0.9}'
OUT_OF_CONTEXT = '{"answer": "Готово", "sources": [5], "confidence": 0.7}'


def collect(service: RAGService, question: str) -> list[StreamEvent]:
    async def run() -> list[StreamEvent]:
        events = [event async for event in service.stream(question)]
        assert isinstance(events[0], RetrievedEvent)
        return events[1:]

    return asyncio.run(run())


def test_answer_cites_sources_from_numbered_context() -> None:
    llm = ScriptedLLM([VALID])

    answer = asyncio.run(make_service(llm, HITS).answer("Какой лимит файла?"))

    assert answer.text == "Лимит — 20 МБ [1]."
    assert answer.confidence == 0.9
    [source] = answer.sources
    assert (source.filename, source.page, source.section) == ("faq.md", 2, "Раздел")
    assert source.quote == "Файлы больше двадцати мегабайт отклоняются."
    assert set(answer.latency_ms) == {"embed", "search", "llm", "total"}
    [request] = llm.requests
    system, user = request.messages
    assert system.content == PROMPT.system
    assert "[1] (файл: faq.md, стр. 2, раздел: Раздел)" in user.content
    assert "[2] (файл: guide.pdf, стр. 7" in user.content
    assert user.content.endswith("Вопрос: Какой лимит файла?")
    assert request.json_schema is not None


def test_invalid_answer_is_retried_with_error_details() -> None:
    llm = ScriptedLLM([OUT_OF_CONTEXT, VALID])

    answer = asyncio.run(make_service(llm, HITS).answer("вопрос"))

    assert answer.text == "Лимит — 20 МБ [1]."
    retry = llm.requests[1].messages
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == OUT_OF_CONTEXT
    assert "вне контекста" in retry[3].content


def test_falls_back_after_second_invalid_answer() -> None:
    llm = ScriptedLLM(["не JSON", OUT_OF_CONTEXT])

    answer = asyncio.run(make_service(llm, HITS).answer("вопрос"))

    assert answer.text == NO_ANSWER
    assert answer.sources == []
    assert answer.confidence == 0.0
    assert len(llm.requests) == 2


def test_low_scores_skip_llm_and_refuse() -> None:
    llm = ScriptedLLM([VALID])
    service = make_service(llm, [hit("шум", score=0.1)], RAGSettings(min_score=0.5))

    answer = asyncio.run(service.answer("Сколько стоит билет на Марс?"))

    assert answer.text == NO_ANSWER
    assert llm.requests == []
    assert "llm" not in answer.latency_ms


def test_answer_without_sources_becomes_refusal() -> None:
    llm = ScriptedLLM(['{"answer": "Думаю, 42.", "sources": [], "confidence": 0.8}'])

    answer = asyncio.run(make_service(llm, HITS).answer("вопрос"))

    assert answer.text == NO_ANSWER
    assert answer.confidence == 0.0


def test_duplicate_sources_are_collapsed() -> None:
    llm = ScriptedLLM(['{"answer": "a", "sources": [2, 1, 2], "confidence": 0.5}'])

    answer = asyncio.run(make_service(llm, HITS).answer("вопрос"))

    assert [s.filename for s in answer.sources] == ["guide.pdf", "faq.md"]


def test_context_budget_drops_lowest_ranked_chunks() -> None:
    llm = ScriptedLLM(['{"answer": "a", "sources": [2], "confidence": 0.5}', VALID])
    service = make_service(llm, HITS, RAGSettings(context_tokens=15))

    answer = asyncio.run(service.answer("вопрос"))

    assert "[2]" not in llm.requests[0].messages[1].content
    assert "[2] вне контекста" in llm.requests[1].messages[-1].content
    assert answer.sources[0].filename == "faq.md"


def test_stream_emits_answer_text_then_sources() -> None:
    events = collect(make_service(ScriptedLLM([VALID]), HITS), "вопрос")

    *tokens, final = events
    assert all(isinstance(e, TokenEvent) for e in tokens)
    assert "".join(e.text for e in tokens if isinstance(e, TokenEvent)) == "Лимит — 20 МБ [1]."
    assert isinstance(final, FinalEvent)
    assert [s.filename for s in final.sources] == ["faq.md"]
    assert final.confidence == 0.9
    assert {"first_token", "llm", "total"} <= set(final.latency_ms)


def test_stream_with_invalid_json_ends_without_sources() -> None:
    events = collect(make_service(ScriptedLLM([OUT_OF_CONTEXT]), HITS), "вопрос")

    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.sources == []
    assert final.confidence == 0.0


def test_stream_without_relevant_chunks_refuses() -> None:
    llm = ScriptedLLM([VALID])

    events = collect(make_service(llm, [], RAGSettings(min_score=0.5)), "вопрос")

    assert events[0] == TokenEvent(NO_ANSWER)
    assert isinstance(events[1], FinalEvent)
    assert llm.requests == []


def test_info_log_has_metadata_but_not_question() -> None:
    with structlog.testing.capture_logs() as logs:
        asyncio.run(make_service(ScriptedLLM([VALID]), HITS).answer("секретный вопрос"))

    [entry] = [e for e in logs if e["event"] == "rag_answer"]
    assert entry["log_level"] == "info"
    assert entry["provider"] == "scripted"
    assert entry["prompt_version"] == "answer/v1"
    assert entry["chunk_ids"] == [str(h.chunk_id) for h in HITS]
    assert (entry["prompt_tokens"], entry["completion_tokens"]) == (100, 20)
    assert "секретный вопрос" not in str(entry)
    assert all(e["log_level"] == "debug" for e in logs if "секретный вопрос" in str(e))
