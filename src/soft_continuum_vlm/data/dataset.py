from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soft_continuum_vlm.data.serialization import load_demo_npz

try:  # pragma: no cover - exercised when torch is installed
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover
    TorchDataset = object


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


class DemoDataset(TorchDataset):  # type: ignore[misc]
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data = load_demo_npz(self.path)

    def __len__(self) -> int:
        return int(self._data["action_vector"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {}
        for key, value in self._data.items():
            element = value[index]
            if isinstance(element, np.generic):
                item[key] = element.item()
            else:
                item[key] = element
        return item

    def action_dim(self) -> int:
        return int(self._data["action_vector"].shape[1])

    def proprioception_dim(self) -> int:
        return int(self._data["proprioception"].shape[1])

    def contact_dim(self) -> int:
        return int(self._data["contact"].shape[1])

    def morphology_dim(self) -> int:
        return int(self._data["morphology"].shape[1])
