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
        previous_action: Mapping[str, Any] | None = None,
        current_robot_state: Mapping[str, Any] | None = None,
        safety_mode: str = "drop_blocked_fields",
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
            if safety_mode == "drop_blocked_fields":
                for field in blocked_fields:
                    safe_action.pop(field, None)
            elif safety_mode == "hold_current":
                self._hold_current(safe_action, previous_action, current_robot_state)
            elif safety_mode == "scale_down":
                self._scale_down(safe_action, current_robot_state)
            else:
                raise ValueError(
                    "safety_mode must be one of 'drop_blocked_fields', 'hold_current', or 'scale_down'."
                )
        return safe_action, {
            "clipped": bool(clipped_fields or blocked_fields),
            "clipped_fields": clipped_fields,
            "blocked_fields": blocked_fields,
            "contact_limited": contact_limited,
            "penetration_limited": penetration_limited,
            "safety_mode": safety_mode,
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

    @classmethod
    def _hold_current(
        cls,
        safe_action: dict[str, Any],
        previous_action: Mapping[str, Any] | None,
        current_robot_state: Mapping[str, Any] | None,
    ) -> None:
        source = current_robot_state or previous_action or {}
        if "section_angles" in safe_action and "section_angles" in source:
            safe_action["section_angles"] = cls._as_float_sequence(source["section_angles"])
        if "grasper_rotation" in safe_action and "grasper_rotation" in source:
            safe_action["grasper_rotation"] = float(source["grasper_rotation"])

    @classmethod
    def _scale_down(
        cls,
        safe_action: dict[str, Any],
        current_robot_state: Mapping[str, Any] | None,
    ) -> None:
        source = current_robot_state or {}
        if "section_angles" in safe_action:
            target = cls._as_float_sequence(safe_action["section_angles"])
            current = cls._as_float_sequence(source.get("section_angles", [0.0] * len(target)))
            safe_action["section_angles"] = [c + 0.2 * (t - c) for c, t in zip(current, target)]
        if "grasper_rotation" in safe_action:
            current_rotation = float(source.get("grasper_rotation", 0.0))
            target_rotation = float(safe_action["grasper_rotation"])
            safe_action["grasper_rotation"] = current_rotation + 0.2 * (target_rotation - current_rotation)
