from dataclasses import dataclass
from pathlib import Path
from string import Template

from app.adapters.llm.base import ChatMessage


@dataclass(frozen=True)
class AnswerPrompt:
    version: str
    system: str
    user: Template
    retry: Template

    def messages(self, question: str, context: str) -> tuple[ChatMessage, ...]:
        return (
            ChatMessage("system", self.system),
            ChatMessage("user", self.user.substitute(question=question, context=context)),
        )

    def retry_messages(
        self, messages: tuple[ChatMessage, ...], previous: str, errors: str
    ) -> tuple[ChatMessage, ...]:
        return (
            *messages,
            ChatMessage("assistant", previous),
            ChatMessage("user", self.retry.substitute(errors=errors)),
        )


def load_answer_prompt(prompts_dir: Path, version: str) -> AnswerPrompt:
    folder = prompts_dir / "answer" / version
    if not folder.is_dir():
        raise FileNotFoundError(f"нет промпта answer версии {version!r} в {prompts_dir}")

    def read(name: str) -> str:
        return (folder / f"{name}.md").read_text(encoding="utf-8").strip()

    return AnswerPrompt(
        version=f"answer/{version}",
        system=read("system"),
        user=Template(read("user")),
        retry=Template(read("retry")),
    )
