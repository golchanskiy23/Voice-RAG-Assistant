import asyncio
import contextlib
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.adapters.llm.base import (
    ChatMessage,
    Completion,
    LLMChunk,
    LLMError,
    LLMRequest,
    StreamEnd,
    TextDelta,
    Usage,
)

Effort = Literal["low", "medium", "high", "xhigh", "max"]
ROLE_TAGS = {"user": "пользователь", "assistant": "ассистент"}


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class ProcessRunner(Protocol):
    async def __call__(self, args: Sequence[str], stdin: bytes, cwd: str) -> ProcessResult: ...


async def run_process(args: Sequence[str], stdin: bytes, cwd: str) -> ProcessResult:
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate(stdin)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise
    return ProcessResult(process.returncode or 0, stdout, stderr)


def render_prompt(messages: Sequence[ChatMessage]) -> str:
    turns = [m for m in messages if m.role != "system"]
    if not turns:
        return ""
    *history, current = turns
    if not history:
        return current.content
    blocks = [f"<{ROLE_TAGS[m.role]}>\n{m.content}\n</{ROLE_TAGS[m.role]}>" for m in history]
    return (
        "Предыдущие реплики:\n\n"
        + "\n\n".join(blocks)
        + "\n\nОтветь на последнюю реплику пользователя:\n\n"
        + current.content
    )


def system_prompt(messages: Sequence[ChatMessage]) -> str:
    return "\n\n".join(m.content for m in messages if m.role == "system")


def parse_usage(payload: dict[str, Any]) -> Usage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = sum(
        int(usage.get(key) or 0)
        for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return Usage(prompt, int(usage.get("output_tokens") or 0))


class ClaudeCodeLLMClient:
    def __init__(
        self,
        binary: str = "claude",
        model: str = "sonnet",
        effort: Effort = "low",
        timeout: float = 120.0,
        concurrency: int = 2,
        runner: ProcessRunner = run_process,
    ) -> None:
        self._binary = binary
        self._model = model
        self._effort = effort
        self._timeout = timeout
        self._slots = asyncio.Semaphore(concurrency)
        self._runner = runner
        self._workdir = tempfile.mkdtemp(prefix="claude-code-llm-")

    @property
    def name(self) -> str:
        return "claude-code"

    def command(self, request: LLMRequest) -> list[str]:
        args = [
            self._binary,
            "--print",
            "--output-format",
            "json",
            "--model",
            self._model,
            "--effort",
            self._effort,
            "--tools",
            "",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--system-prompt",
            system_prompt(request.messages),
        ]
        if request.json_schema is not None:
            args += ["--json-schema", json.dumps(request.json_schema.schema, ensure_ascii=False)]
        return args

    async def complete(self, request: LLMRequest) -> Completion:
        prompt = render_prompt(request.messages).encode()
        try:
            async with self._slots, asyncio.timeout(self._timeout):
                result = await self._runner(self.command(request), prompt, self._workdir)
        except TimeoutError as exc:
            raise LLMError(self.name, f"нет ответа за {self._timeout:.0f} с") from exc
        except OSError as exc:
            raise LLMError(self.name, f"не удалось запустить {self._binary}: {exc}") from exc
        return self._completion(result)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        completion = await self.complete(request)
        yield TextDelta(completion.text)
        yield StreamEnd(provider=self.name, usage=completion.usage)

    async def is_ready(self) -> bool:
        return shutil.which(self._binary) is not None

    async def close(self) -> None:
        await asyncio.to_thread(shutil.rmtree, self._workdir, True)

    def _completion(self, result: ProcessResult) -> Completion:
        try:
            payload = json.loads(result.stdout)
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            details = result.stderr.decode(errors="replace").strip()[-300:]
            raise LLMError(self.name, f"код {result.returncode}: {details or 'пустой ответ'}")
        if payload.get("is_error") or payload.get("subtype") != "success":
            details = str(payload.get("result") or payload.get("subtype") or "ошибка")[:300]
            raise LLMError(self.name, details)
        structured = payload.get("structured_output")
        text = (
            json.dumps(structured, ensure_ascii=False)
            if structured is not None
            else str(payload.get("result") or "")
        )
        return Completion(text=text, provider=self.name, usage=parse_usage(payload))
