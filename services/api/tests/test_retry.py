import asyncio

import pytest

from app.core.retry import retry_async


class Flaky:
    def __init__(self, failures: int, error: type[Exception] = ConnectionError) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("временная ошибка")
        return "ok"


def run(operation: Flaky, delays: list[float], attempts: int = 4) -> str:
    async def sleep(delay: float) -> None:
        delays.append(delay)

    return asyncio.run(
        retry_async(
            operation,
            retry_on=(ConnectionError,),
            attempts=attempts,
            base_delay=1.0,
            max_delay=5.0,
            sleep=sleep,
        )
    )


def test_succeeds_after_transient_failures_with_bounded_backoff() -> None:
    delays: list[float] = []
    operation = Flaky(failures=3)

    assert run(operation, delays) == "ok"
    assert operation.calls == 4
    assert len(delays) == 3
    assert all(0 <= d <= cap for d, cap in zip(delays, [1.0, 2.0, 4.0], strict=True))


def test_gives_up_after_last_attempt() -> None:
    delays: list[float] = []
    operation = Flaky(failures=10)

    with pytest.raises(ConnectionError):
        run(operation, delays, attempts=3)
    assert operation.calls == 3
    assert len(delays) == 2


def test_does_not_retry_other_errors() -> None:
    delays: list[float] = []
    operation = Flaky(failures=1, error=ValueError)

    with pytest.raises(ValueError):
        run(operation, delays)
    assert operation.calls == 1
    assert delays == []


def test_delay_is_capped_by_max_delay() -> None:
    delays: list[float] = []

    run(Flaky(failures=7), delays, attempts=8)

    assert max(delays) <= 5.0
