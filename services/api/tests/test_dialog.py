import asyncio
import json
from collections.abc import Iterator

import fakeredis
import pytest

from app.adapters.dialog.base import DialogStore, Turn
from app.adapters.dialog.memory import InMemoryDialogStore
from app.adapters.dialog.redis import RedisDialogStore
from app.adapters.llm.base import Completion, LLMError, LLMRequest
from app.core.config import PROMPTS_DIR
from app.services.dialog import DialogService, QueryRewriter, load_rewrite_prompt
from tests.llm_fakes import ScriptedLLM, hit, make_service

REWRITE_PROMPT = load_rewrite_prompt(PROMPTS_DIR)


@pytest.fixture(params=["memory", "redis"])
def store(request: pytest.FixtureRequest) -> Iterator[DialogStore]:
    if request.param == "memory":
        yield InMemoryDialogStore(max_turns=4)
    else:
        yield RedisDialogStore(fakeredis.FakeAsyncRedis(decode_responses=True), max_turns=4)


def test_history_is_trimmed_to_last_turns(store: DialogStore) -> None:
    async def run() -> list[Turn]:
        for i in range(3):
            await store.append("s", [Turn("user", f"q{i}"), Turn("assistant", f"a{i}")])
        return await store.history("s")

    assert [t.text for t in asyncio.run(run())] == ["q1", "a1", "q2", "a2"]


def test_clear_and_isolation(store: DialogStore) -> None:
    async def run() -> tuple[list[Turn], list[Turn]]:
        await store.append("a", [Turn("user", "x")])
        await store.append("b", [Turn("user", "y")])
        await store.clear("a")
        return await store.history("a"), await store.history("b")

    cleared, other = asyncio.run(run())
    assert cleared == [] and [t.text for t in other] == ["y"]


def test_first_seen_deduplicates(store: DialogStore) -> None:
    async def run() -> list[bool]:
        return [await store.first_seen("upd:1", 60) for _ in range(2)] + [
            await store.first_seen("upd:2", 60)
        ]

    assert asyncio.run(run()) == [True, False, True]


def test_rate_limit_per_key(store: DialogStore) -> None:
    async def run() -> tuple[list[bool], bool]:
        hits = [await store.hit_rate_limit("u1", 3, 60) for _ in range(4)]
        return hits, await store.hit_rate_limit("u2", 3, 60)

    hits, other = asyncio.run(run())
    assert hits == [False, False, False, True]
    assert other is False


def test_redis_history_has_ttl() -> None:
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    store = RedisDialogStore(redis, ttl_seconds=100)

    async def run() -> int:
        await store.append("s", [Turn("user", "x")])
        return int(await redis.ttl("dialog:s"))

    assert 0 < asyncio.run(run()) <= 100


class RewritingLLM(ScriptedLLM):
    async def complete(self, request: LLMRequest) -> Completion:
        self.requests.append(request)
        if request.json_schema and request.json_schema.name == "rewritten_question":
            return Completion(
                json.dumps(
                    {"question": "Сколько стоит ДМС для родственников?"}, ensure_ascii=False
                ),
                "scripted",
            )
        return Completion(self._responses[0], "scripted")


ANSWER = '{"answer": "За свой счёт по корпоративному тарифу.", "sources": [1], "confidence": 0.8}'


def test_follow_up_question_is_rewritten_before_search() -> None:
    llm = RewritingLLM([ANSWER])
    store = InMemoryDialogStore()
    service = DialogService(
        store, QueryRewriter(llm, REWRITE_PROMPT), make_service(llm, [hit("ДМС")])
    )

    async def run() -> None:
        await store.append(
            "s", [Turn("user", "Можно подключить родственников к ДМС?"), Turn("assistant", "Да.")]
        )
        result = await service.answer("s", "а сколько это стоит?")
        assert result.standalone_question == "Сколько стоит ДМС для родственников?"
        assert [t.text for t in await store.history("s")][-2:] == [
            "а сколько это стоит?",
            "За свой счёт по корпоративному тарифу.",
        ]

    asyncio.run(run())
    rewrite, answer = llm.requests
    assert "Пользователь: Можно подключить родственников к ДМС?" in rewrite.messages[1].content
    assert rewrite.messages[1].content.endswith("Вопрос: а сколько это стоит?")
    assert "Вопрос: Сколько стоит ДМС для родственников?" in answer.messages[-1].content


def test_first_question_skips_rewrite() -> None:
    llm = RewritingLLM([ANSWER])
    service = DialogService(
        InMemoryDialogStore(), QueryRewriter(llm, REWRITE_PROMPT), make_service(llm, [hit("x")])
    )

    result = asyncio.run(service.answer("s", "Что такое ДМС?"))

    assert result.standalone_question == "Что такое ДМС?"
    assert len(llm.requests) == 1


class BrokenRewriteLLM(ScriptedLLM):
    async def complete(self, request: LLMRequest) -> Completion:
        raise LLMError("broken", "down")


def test_rewrite_failure_falls_back_to_original_question() -> None:
    rewriter = QueryRewriter(BrokenRewriteLLM([]), REWRITE_PROMPT)

    rewritten = asyncio.run(rewriter.rewrite("а это?", [Turn("user", "x")]))

    assert rewritten == "а это?"


def test_rewrite_prompt_rules() -> None:
    assert '{"question"' in REWRITE_PROMPT.system
    assert "не инструкции" in REWRITE_PROMPT.system
    _, user = REWRITE_PROMPT.messages("а $x?", [Turn("user", "<тег>")])
    assert "‹тег›" in user.content and user.content.endswith("Вопрос: а $x?")
