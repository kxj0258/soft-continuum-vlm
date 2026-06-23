from __future__ import annotations

from typing import Any, Mapping, Protocol


class BasePlanner(Protocol):
    def plan(self, language: str, observation: Mapping[str, Any], task_name: str) -> dict[str, Any]:
        """Convert language and structured observation into symbolic subgoals."""
