from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Mapping


ACTION_FIELDS = (
    "section_angles",
    "grip_command",
    "grasper_rotation",
)


@dataclass(frozen=True)
class SafetyLimits:
    max_abs_section_angle: float
    max_gripper_rotation: float
    max_contact_force: float
    max_penetration: float


class SafetyProjector:
    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    def project(
        self,
        action: Mapping[str, Any],
        *,
        contact_force: float = 0.0,
        penetration: float = 0.0,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        safe_action: dict[str, Any] = {}
        clipped_fields: list[str] = []
        if "section_angles" in action:
            section_angles = self._as_float_sequence(action["section_angles"])
            safe_section_angles = [
                self._clip(value, self.limits.max_abs_section_angle) for value in section_angles
            ]
            safe_action["section_angles"] = safe_section_angles
            if safe_section_angles != section_angles:
                clipped_fields.append("section_angles")
        if "grip_command" in action:
            grip_command = min(max(float(action["grip_command"]), 0.0), 1.0)
            safe_action["grip_command"] = grip_command
            if grip_command != float(action["grip_command"]):
                clipped_fields.append("grip_command")
        if "grasper_rotation" in action:
            grasper_rotation = self._clip(action["grasper_rotation"], self.limits.max_gripper_rotation)
            safe_action["grasper_rotation"] = grasper_rotation
            if grasper_rotation != float(action["grasper_rotation"]):
                clipped_fields.append("grasper_rotation")
        for passthrough_field in ("joint_targets", "segment_joint_targets"):
            if passthrough_field in action:
                safe_action[passthrough_field] = action[passthrough_field]

        contact_limited = contact_force > self.limits.max_contact_force
        penetration_limited = penetration > self.limits.max_penetration
        blocked_fields: list[str] = []
        if contact_limited or penetration_limited:
            for field in ("section_angles", "grasper_rotation", "joint_targets", "segment_joint_targets"):
                if field in safe_action:
                    blocked_fields.append(field)
                    safe_action.pop(field)
        return safe_action, {
            "clipped": bool(clipped_fields or blocked_fields),
            "clipped_fields": clipped_fields,
            "blocked_fields": blocked_fields,
            "contact_limited": contact_limited,
            "penetration_limited": penetration_limited,
        }

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        numeric_value = float(value)
        return min(max(numeric_value, -limit), limit)

    @staticmethod
    def _as_float_sequence(value: Any) -> list[float]:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise TypeError("section_angles must be a numeric sequence.")
        return [float(item) for item in value]
