from __future__ import annotations

from typing import Any, Mapping, Protocol


class PolicyProtocol(Protocol):
    baseline_name: str

    def reset(self, task_name: str, language: str | None = None) -> None:
        ...

    def act(self, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        ...
