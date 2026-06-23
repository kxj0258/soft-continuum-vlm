from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskSpec:
    name: str
    language: str
    target_object: str = ""
    success: Mapping[str, Any] = field(default_factory=dict)


class BaseTask:
    """Deterministic task interface used before full simulator integration."""

    spec: TaskSpec

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def language(self) -> str:
        return self.spec.language

    def reset_metadata(self) -> dict[str, Any]:
        return {
            "task": self.spec.name,
            "language": self.spec.language,
            "target_object": self.spec.target_object,
        }

    def evaluate(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "success": False,
            "metrics": {},
            "observation_keys": sorted(observation.keys()),
            "deferred_note": (
                "Expected input: Feagine observation dict. Expected output: task "
                "success flag and metrics. Integration path: bind task-specific "
                "geometry and contact checks after Milestone 2."
            ),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "BaseTask":
        raw = dict(config.get("task", config))
        return cls(
            TaskSpec(
                name=str(raw["name"]),
                language=str(raw.get("language", "")),
                target_object=str(raw.get("target_object", "")),
                success=dict(raw.get("success", {})),
            )
        )
