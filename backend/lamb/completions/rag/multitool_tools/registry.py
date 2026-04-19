from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List

ToolFn = Callable[..., Awaitable[Dict[str, Any]]]


class ToolRegistry:
    """Maps tool names to async callables.

    Single source of truth for valid tool names at runtime.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def get(self, name: str) -> ToolFn:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def allowed_names(self) -> List[str]:
        return sorted(self._tools.keys())
