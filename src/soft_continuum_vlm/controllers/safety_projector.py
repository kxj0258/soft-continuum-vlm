from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ACTION_FIELDS = (
    "delta_kappa_x",
    "delta_kappa_y",
    "delta_length",
    "gripper_open",
    "gripper_rotation",
)


@dataclass(frozen=True)
class SafetyLimits:
    max_delta_kappa: float
    max_delta_length: float
    max_gripper_rotation: float
    max_contact_force: float
    max_penetration: float


class SafetyProjector:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    def project(
        self,
        action: Mapping[str, float],
        *,
        contact_force: float = 0.0,
        penetration: float = 0.0,
    ) -> tuple[dict[str, float], dict[str, object]]:
        safe_action = {
            "delta_kappa_x": self._clip(action.get("delta_kappa_x", 0.0), self.limits.max_delta_kappa),
            "delta_kappa_y": self._clip(action.get("delta_kappa_y", 0.0), self.limits.max_delta_kappa),
            "delta_length": self._clip(action.get("delta_length", 0.0), self.limits.max_delta_length),
            "gripper_open": min(max(float(action.get("gripper_open", 0.0)), 0.0), 1.0),
            "gripper_rotation": self._clip(
                action.get("gripper_rotation", 0.0), self.limits.max_gripper_rotation
            ),
        }
        clipped_fields = [
            field for field in ACTION_FIELDS if safe_action[field] != float(action.get(field, 0.0))
        ]
        contact_limited = contact_force > self.limits.max_contact_force
        penetration_limited = penetration > self.limits.max_penetration
        if contact_limited or penetration_limited:
            for field in ("delta_kappa_x", "delta_kappa_y", "delta_length", "gripper_rotation"):
                if safe_action[field] != 0.0 and field not in clipped_fields:
                    clipped_fields.append(field)
                safe_action[field] = 0.0
        return safe_action, {
            "clipped": bool(clipped_fields),
            "clipped_fields": clipped_fields,
            "contact_limited": contact_limited,
            "penetration_limited": penetration_limited,
        }

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        numeric_value = float(value)
        return min(max(numeric_value, -limit), limit)
