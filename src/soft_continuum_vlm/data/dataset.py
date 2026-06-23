from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DemoTransition:
    observation: dict[str, Any]
    action: dict[str, float]
    next_observation: dict[str, Any]
    done: bool
    info: dict[str, Any]


class InMemoryDemoDataset(Sequence[DemoTransition]):
    def __init__(self, transitions: Sequence[DemoTransition]) -> None:
        self._transitions = tuple(transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, index: int) -> DemoTransition:
        return self._transitions[index]
