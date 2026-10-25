from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator

NO_ANSWER = "В документах нет ответа на этот вопрос."


class LLMAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1)
    sources: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def sources_within_context(self, info: ValidationInfo) -> Self:
        chunk_count = (info.context or {}).get("chunk_count", 0)
        unknown = sorted({n for n in self.sources if not 1 <= n <= chunk_count})
        if unknown:
            raise ValueError(f"источники {unknown} вне контекста: доступны номера 1..{chunk_count}")
        return self


class AnswerValidationError(Exception):
    def __init__(self, details: str) -> None:
        super().__init__(details)
        self.details = details


def parse_llm_answer(raw: str, chunk_count: int) -> LLMAnswer:
    try:
        return LLMAnswer.model_validate_json(raw, context={"chunk_count": chunk_count})
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'answer'}: {error['msg']}"
            for error in exc.errors()
        )
        raise AnswerValidationError(details) from exc
