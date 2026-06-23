from __future__ import annotations

from collections import deque
from typing import Deque, Iterable

from soft_continuum_vlm.data.dataset import DemoTransition


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self._items: Deque[DemoTransition] = deque(maxlen=capacity)

    def append(self, transition: DemoTransition) -> None:
        self._items.append(transition)

    def extend(self, transitions: Iterable[DemoTransition]) -> None:
        for transition in transitions:
            self.append(transition)

    def __len__(self) -> int:
        return len(self._items)

    def to_dataset(self) -> list[DemoTransition]:
        return list(self._items)
