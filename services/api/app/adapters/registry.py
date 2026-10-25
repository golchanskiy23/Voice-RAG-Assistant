from collections.abc import Callable


class UnknownAdapterError(LookupError):
    pass


class Registry[T]:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[[], T]] = {}

    @property
    def names(self) -> list[str]:
        return sorted(self._factories)

    def register(self, name: str, factory: Callable[[], T]) -> None:
        if name in self._factories:
            raise ValueError(f"{self._kind}: реализация {name!r} уже зарегистрирована")
        self._factories[name] = factory

    def create(self, name: str) -> T:
        try:
            factory = self._factories[name]
        except KeyError:
            raise UnknownAdapterError(
                f"{self._kind}: неизвестная реализация {name!r}, доступны: {self.names}"
            ) from None
        return factory()
