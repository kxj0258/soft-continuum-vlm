from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.controllers.continuum_kinematics import (
    ContinuumGeometry,
    damped_least_squares_step,
)


def zero_feagine_action(section_count: int) -> dict[str, object]:
    return {
        "section_angles": [0.0] * (2 * section_count),
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    }


@dataclass
class PccIkConfig:
    section_count: int = 3
    section_length: float = 0.10
    max_abs_section_angle: float = 0.8
    max_step_norm: float = 0.25
    position_gain: float = 1.0
    min_target_distance: float = 0.005


class PccIkController:
    """Approximate PCC IK controller producing Feagine section-angle actions."""

    def __init__(self, config: PccIkConfig | None = None, *, section_count: int | None = None) -> None:
        if config is None:
            config = PccIkConfig(section_count=section_count or 3)
        elif section_count is not None:
            config = PccIkConfig(
                section_count=section_count,
                section_length=config.section_length,
                max_abs_section_angle=config.max_abs_section_angle,
                max_step_norm=config.max_step_norm,
                position_gain=config.position_gain,
                min_target_distance=config.min_target_distance,
            )
        self.config = config
        self.section_count = config.section_count
        self.geometry = ContinuumGeometry(
            section_count=config.section_count,
            section_length=config.section_length,
            max_abs_section_angle=config.max_abs_section_angle,
        )

    def compute_action(
        self,
        target_state: Mapping[str, Any],
        robot_state: Mapping[str, Any],
    ) -> dict[str, object]:
        current_angles = self._current_section_angles(robot_state)
        grip_command = float(target_state.get("grip_command", robot_state.get("grip_command", 0.0)))
        grasper_rotation = float(
            target_state.get("grasper_rotation", robot_state.get("grasper_rotation", 0.0))
        )
        target_position = target_state.get("target_tip_position")
        current_position = self._current_tip_position(robot_state)
        if target_position is None or current_position is None:
            return {
                "section_angles": current_angles,
                "grip_command": grip_command,
                "grasper_rotation": grasper_rotation,
            }

        target = np.asarray(target_position, dtype=np.float64).reshape(-1)[:3]
        current = np.asarray(current_position, dtype=np.float64).reshape(-1)[:3]
        error = (target - current) * float(self.config.position_gain)
        distance = float(np.linalg.norm(error))
        if distance <= self.config.min_target_distance:
            next_angles = current_angles
        else:
            desired_angles = np.asarray(
                damped_least_squares_step(current_angles, error, self.geometry),
                dtype=np.float64,
            )
            current_array = np.asarray(current_angles, dtype=np.float64)
            step = desired_angles - current_array
            step_norm = float(np.linalg.norm(step))
            if step_norm > self.config.max_step_norm > 0.0:
                step *= self.config.max_step_norm / step_norm
            next_angles_array = np.clip(
                current_array + step,
                -self.config.max_abs_section_angle,
                self.config.max_abs_section_angle,
            )
            next_angles = [float(value) for value in next_angles_array]
        return {
            "section_angles": next_angles,
            "grip_command": min(max(grip_command, 0.0), 1.0),
            "grasper_rotation": float(
                min(max(grasper_rotation, -self.config.max_abs_section_angle), self.config.max_abs_section_angle)
            ),
        }

    def _current_section_angles(self, robot_state: Mapping[str, Any]) -> list[float]:
        raw = robot_state.get("section_angles")
        if raw is None and "proprioception" in robot_state:
            raw = list(robot_state.get("proprioception", []))[: 2 * self.section_count]
        values = list(raw or [])
        if len(values) != 2 * self.section_count:
            return [0.0] * (2 * self.section_count)
        return [float(value) for value in values]

    @staticmethod
    def _current_tip_position(robot_state: Mapping[str, Any]) -> list[float] | None:
        tip_pose = robot_state.get("tip_pose")
        if isinstance(tip_pose, Mapping) and "position" in tip_pose:
            return [float(value) for value in list(tip_pose["position"])[:3]]
        if "proprioception" in robot_state:
            return [0.0, 0.0, 0.0]
        return None
