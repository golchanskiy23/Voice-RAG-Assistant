from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.adapters.dialog.base import DialogStore, Turn
from app.adapters.llm.base import ChatMessage, JsonSchema, LLMClient, LLMError, LLMRequest
from app.services.prompts import escape_text
from app.services.rag import Answer, RAGService

log = structlog.get_logger()

REWRITE_SCHEMA = JsonSchema(
    name="rewritten_question",
    schema={
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
        "additionalProperties": False,
    },
)
ROLE_LABELS = {"user": "Пользователь", "assistant": "Ассистент"}


class Rewritten(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True)
class RewritePrompt:
    version: str
    system: str
    user: Template

    def messages(self, question: str, history: list[Turn]) -> tuple[ChatMessage, ...]:
        lines = "\n".join(f"{ROLE_LABELS[t.role]}: {escape_text(t.text)}" for t in history)
        return (
            ChatMessage("system", self.system),
            ChatMessage("user", self.user.substitute(history=lines, question=question)),
        )


def load_rewrite_prompt(prompts_dir: Path, version: str = "v1") -> RewritePrompt:
    folder = prompts_dir / "rewrite" / version
    return RewritePrompt(
        version=f"rewrite/{version}",
        system=(folder / "system.md").read_text(encoding="utf-8").strip(),
        user=Template((folder / "user.md").read_text(encoding="utf-8").strip()),
    )


class QueryRewriter:
    def __init__(self, llm: LLMClient, prompt: RewritePrompt, max_tokens: int = 128) -> None:
        self._llm = llm
        self._prompt = prompt
        self._max_tokens = max_tokens

    async def rewrite(self, question: str, history: list[Turn]) -> str:
        if not history:
            return question
        request = LLMRequest(
            messages=self._prompt.messages(question, history),
            json_schema=REWRITE_SCHEMA,
            max_tokens=self._max_tokens,
        )
        try:
            completion = await self._llm.complete(request)
            rewritten = Rewritten.model_validate_json(completion.text).question
        except (LLMError, ValidationError) as exc:
            log.warning("rewrite_failed", stage="rewrite", error=type(exc).__name__)
            return question
        log.info(
            "rewrite",
            stage="rewrite",
            prompt_version=self._prompt.version,
            changed=rewritten != question,
        )
        log.debug("rewrite_text", question=question, rewritten=rewritten)
        return rewritten


class Rewriter(Protocol):
    async def rewrite(self, question: str, history: list[Turn]) -> str: ...


class NoRewrite:
    async def rewrite(self, question: str, history: list[Turn]) -> str:
        return question


@dataclass(frozen=True)
class DialogAnswer:
    answer: Answer
    standalone_question: str


class DialogService:
    def __init__(self, store: DialogStore, rewriter: Rewriter, rag: RAGService) -> None:
        self._store = store
        self._rewriter = rewriter
        self._rag = rag

    async def answer(self, session: str, question: str) -> DialogAnswer:
        history = await self._store.history(session)
        standalone = await self._rewriter.rewrite(question, history)
        answer = await self._rag.answer(standalone)
        await self._store.append(session, [Turn("user", question), Turn("assistant", answer.text)])
        return DialogAnswer(answer, standalone)

    async def reset(self, session: str) -> None:
        await self._store.clear(session)
