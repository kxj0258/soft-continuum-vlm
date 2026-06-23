from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectedObject:
    name: str
    label: str
    position: np.ndarray
    confidence: float = 1.0


@dataclass(frozen=True)
class SceneState:
    objects: tuple[DetectedObject, ...] = field(default_factory=tuple)
    language: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
