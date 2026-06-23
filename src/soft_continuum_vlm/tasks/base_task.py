from __future__ import annotations

from dataclasses import dataclass, field
import math
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

    def target_object(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        objects = observation.get("objects", {})
        if not isinstance(objects, Mapping):
            return {}
        target = objects.get(self.spec.target_object, {})
        return target if isinstance(target, Mapping) else {}

    def target_pose(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        target = self.target_object(observation)
        pose = target.get("pose", {})
        return pose if isinstance(pose, Mapping) else {}

    @staticmethod
    def vector3(
        value: Any,
        default: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return default
        return float(value[0]), float(value[1]), float(value[2])

    @staticmethod
    def quaternion(value: Any) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return 1.0, 0.0, 0.0, 0.0
        return float(value[0]), float(value[1]), float(value[2]), float(value[3])

    @classmethod
    def position_error(cls, first: Any, second: Any) -> float:
        ax, ay, az = cls.vector3(first)
        bx, by, bz = cls.vector3(second)
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    @classmethod
    def orientation_error(cls, first: Any, second: Any) -> float:
        qa = cls.quaternion(first)
        qb = cls.quaternion(second)
        dot = abs(sum(a * b for a, b in zip(qa, qb)))
        dot = min(max(dot, -1.0), 1.0)
        return 2.0 * math.acos(dot)

    @classmethod
    def contact_force_norm(cls, contact: Mapping[str, Any]) -> float:
        force = contact.get("force")
        if isinstance(force, (list, tuple)) and len(force) >= 3:
            fx, fy, fz = cls.vector3(force)
            return math.sqrt(fx * fx + fy * fy + fz * fz)
        return float(contact.get("force_norm", 0.0))

    def contact_force_for(self, observation: Mapping[str, Any], name: str) -> float:
        contact = observation.get("contact", {})
        if not isinstance(contact, Mapping):
            return 0.0
        contacts = contact.get("contacts", [])
        if not isinstance(contacts, list):
            return 0.0
        max_force = 0.0
        for item in contacts:
            if not isinstance(item, Mapping):
                continue
            geom1 = str(item.get("geom1", ""))
            geom2 = str(item.get("geom2", ""))
            if name in (geom1, geom2):
                max_force = max(max_force, self.contact_force_norm(item))
        return max_force

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
