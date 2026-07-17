NO_ANSWER = "В документах нет ответа на этот вопрос."


class StubLLMClient:
    def __init__(self) -> None:
        self._closed = False

    @property
    def name(self) -> str:
        return "stub"

    async def complete(self, prompt: str) -> str:
        return NO_ANSWER

    async def is_ready(self) -> bool:
        return not self._closed

    async def close(self) -> None:
        self._closed = True
