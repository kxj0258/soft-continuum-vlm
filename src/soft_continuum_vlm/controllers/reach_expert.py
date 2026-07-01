from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.envs.action_space import DEFAULT_DELTA_XYZ_SCALE


class FeagineReachExpert:
    """Deterministic proportional expert that emits only the 4D top-level action."""

    def __init__(self, *, delta_xyz_scale: float = DEFAULT_DELTA_XYZ_SCALE) -> None:
        if not np.isfinite(delta_xyz_scale) or delta_xyz_scale <= 0.0:
            raise ValueError("delta_xyz_scale must be finite and positive.")
        self.delta_xyz_scale = float(delta_xyz_scale)

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        tip = self._tip_position(observation)
        task = observation.get("task", {})
        if not isinstance(task, Mapping) or "goal_position" not in task:
            raise ValueError("observation.task.goal_position is required.")
        goal = self._point3(task["goal_position"], "goal_position")
        normalized_delta = np.clip(
            (goal - tip) / self.delta_xyz_scale,
            -1.0,
            1.0,
        )
        return np.asarray([*normalized_delta, -1.0], dtype=np.float32)

    @classmethod
    def _tip_position(cls, observation: Mapping[str, Any]) -> np.ndarray:
        robot_state = observation.get("robot_state", {})
        if not isinstance(robot_state, Mapping):
            raise ValueError("observation.robot_state must be a mapping.")
        tip_pose = robot_state.get("tip_pose", {})
        if not isinstance(tip_pose, Mapping) or "position" not in tip_pose:
            raise ValueError("observation.robot_state.tip_pose.position is required.")
        return cls._point3(tip_pose["position"], "tip_pose.position")

    @staticmethod
    def _point3(value: Any, label: str) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{label} must contain exactly three finite values.")
        return point
