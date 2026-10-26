import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from app.adapters.llm.base import ChatMessage, LLMError, LLMRequest, StreamEnd, TextDelta
from app.adapters.llm.claude_code import ClaudeCodeLLMClient, ProcessResult, render_prompt
from app.adapters.llm.registry import build_llm
from app.services.llm_answer import ANSWER_SCHEMA
from tests.factories import make_settings

REQUEST = LLMRequest(
    messages=(
        ChatMessage("system", "Отвечай по фрагментам."),
        ChatMessage("user", "Пример вопроса"),
        ChatMessage("assistant", '{"answer": "пример"}'),
        ChatMessage("user", "Какой лимит?"),
    ),
    json_schema=ANSWER_SCHEMA,
)


def success(**extra: Any) -> bytes:
    payload = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "",
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 50,
            "output_tokens": 20,
        },
        **extra,
    }
    return json.dumps(payload, ensure_ascii=False).encode()


class FakeRunner:
    def __init__(self, result: ProcessResult | BaseException) -> None:
        self.result = result
        self.calls: list[tuple[list[str], str]] = []

    async def __call__(self, args: Sequence[str], stdin: bytes, cwd: str) -> ProcessResult:
        self.calls.append((list(args), stdin.decode()))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def client(runner: FakeRunner) -> ClaudeCodeLLMClient:
    return ClaudeCodeLLMClient(model="haiku", runner=runner)


def test_complete_returns_structured_output_and_usage() -> None:
    answer = {"answer": "20 МБ", "sources": [1], "confidence": 0.9}
    runner = FakeRunner(ProcessResult(0, success(structured_output=answer), b""))

    completion = asyncio.run(client(runner).complete(REQUEST))

    assert json.loads(completion.text) == answer
    assert completion.provider == "claude-code"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 152
    assert completion.usage.completion_tokens == 20


def test_command_disables_tools_and_passes_schema_and_system() -> None:
    runner = FakeRunner(ProcessResult(0, success(result="текст"), b""))

    asyncio.run(client(runner).complete(REQUEST))

    [(args, stdin)] = runner.calls
    assert args[:2] == ["claude", "--print"]
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--model") + 1] == "haiku"
    assert args[args.index("--system-prompt") + 1] == "Отвечай по фрагментам."
    assert json.loads(args[args.index("--json-schema") + 1]) == ANSWER_SCHEMA.schema
    assert "--no-session-persistence" in args
    assert stdin.endswith("Какой лимит?")
    assert "<ассистент>" in stdin


def test_plain_result_without_schema() -> None:
    runner = FakeRunner(ProcessResult(0, success(result="привет"), b""))
    request = LLMRequest(messages=(ChatMessage("user", "hi"),))

    completion = asyncio.run(client(runner).complete(request))

    assert completion.text == "привет"
    assert "--json-schema" not in runner.calls[0][0]


@pytest.mark.parametrize(
    "result",
    [
        ProcessResult(1, b"", b"Not logged in"),
        ProcessResult(0, success(is_error=True, subtype="error", result="rate limit"), b""),
        TimeoutError(),
        FileNotFoundError("claude"),
    ],
)
def test_failures_become_llm_error(result: ProcessResult | BaseException) -> None:
    with pytest.raises(LLMError):
        asyncio.run(client(FakeRunner(result)).complete(REQUEST))


def test_stream_yields_whole_completion() -> None:
    runner = FakeRunner(ProcessResult(0, success(result="ответ"), b""))

    async def collect() -> list[object]:
        return [chunk async for chunk in client(runner).stream(REQUEST)]

    chunks = asyncio.run(collect())

    assert chunks[0] == TextDelta("ответ")
    assert isinstance(chunks[1], StreamEnd)
    assert chunks[1].provider == "claude-code"


def test_single_user_message_is_sent_as_is() -> None:
    messages = (ChatMessage("system", "s"), ChatMessage("user", "вопрос"))

    assert render_prompt(messages) == "вопрос"


def test_registry_builds_claude_code() -> None:
    settings = make_settings().model_copy(update={"llm_provider": "claude-code"})

    assert build_llm(settings).name == "claude-code"
