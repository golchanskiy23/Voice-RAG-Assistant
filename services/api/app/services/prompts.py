import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template

from app.adapters.llm.base import ChatMessage
from app.adapters.vector_stores.base import SearchHit

ANGLE_BRACKETS = str.maketrans({"<": "‹", ">": "›"})


def escape_text(text: str) -> str:
    return text.strip().translate(ANGLE_BRACKETS)


def escape_attribute(value: str) -> str:
    return escape_text(value).replace('"', "″")


def chunk_meta(hit: SearchHit) -> str:
    parts = [f"файл: {hit.filename}"]
    if hit.page is not None:
        parts.append(f"стр. {hit.page}")
    if hit.section:
        parts.append(f"раздел: {hit.section}")
    return ", ".join(parts)


def chunk_attributes(hit: SearchHit) -> str:
    attributes = {"файл": hit.filename, "стр": hit.page, "раздел": hit.section}
    return " ".join(
        f'{name}="{escape_attribute(str(value))}"'
        for name, value in attributes.items()
        if value is not None and value != ""
    )


@dataclass(frozen=True)
class Example:
    question: str
    hits: tuple[SearchHit, ...]
    answer: str


@dataclass(frozen=True)
class AnswerPrompt:
    version: str
    system: str
    user: Template
    retry: Template
    chunk: Template
    examples: tuple[Example, ...] = ()

    def render_chunk(self, number: int, hit: SearchHit) -> str:
        return self.chunk.substitute(
            number=number,
            meta=chunk_meta(hit),
            attrs=chunk_attributes(hit),
            text=escape_text(hit.text),
        )

    def context(self, hits: Sequence[SearchHit]) -> str:
        return "\n\n".join(self.render_chunk(n, hit) for n, hit in enumerate(hits, start=1))

    def render_user(self, question: str, hits: Sequence[SearchHit]) -> str:
        return self.user.substitute(question=question, context=self.context(hits))

    def messages(self, question: str, hits: Sequence[SearchHit]) -> tuple[ChatMessage, ...]:
        shots: list[ChatMessage] = []
        for example in self.examples:
            shots.append(ChatMessage("user", self.render_user(example.question, example.hits)))
            shots.append(ChatMessage("assistant", example.answer))
        return (
            ChatMessage("system", self.system),
            *shots,
            ChatMessage("user", self.render_user(question, hits)),
        )

    def retry_messages(
        self, messages: tuple[ChatMessage, ...], previous: str, errors: str
    ) -> tuple[ChatMessage, ...]:
        return (
            *messages,
            ChatMessage("assistant", previous),
            ChatMessage("user", self.retry.substitute(errors=errors)),
        )


def example_hit(chunk: dict[str, str]) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.UUID(int=0),
        document_id=uuid.UUID(int=0),
        filename=chunk["file"],
        ord=0,
        text=chunk["text"],
        page=None,
        section=chunk.get("section"),
        score=1.0,
    )


def load_examples(path: Path) -> tuple[Example, ...]:
    if not path.exists():
        return ()
    return tuple(
        Example(
            question=item["question"],
            hits=tuple(example_hit(chunk) for chunk in item["chunks"]),
            answer=json.dumps(item["answer"], ensure_ascii=False),
        )
        for item in json.loads(path.read_text(encoding="utf-8"))
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
        chunk=Template(read("chunk")),
        examples=load_examples(folder / "examples.json"),
    )
