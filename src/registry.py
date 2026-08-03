"""Generic name-to-class registry shared by every pluggable pipeline stage.

Each stage package (tokenizer_surgery, embedding_init, distillation, ...) owns one
Registry instance and decorates its strategy classes with it. This is the mechanism
that lets a new strategy (e.g. FOCUS embedding init) be added by writing one new file
and registering a name — no changes anywhere else in the pipeline.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str):
        self._kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self._kind} strategy '{name}' is already registered")
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "(none registered)"
            raise KeyError(f"Unknown {self._kind} strategy '{name}'. Available: {available}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)
