from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.envs.action_space import DEFAULT_DELTA_XYZ_SCALE


class FeaginePushExpert:
    """Deterministic two-phase push expert using only the top-level 4D action."""

    def __init__(
        self,
        *,
        delta_xyz_scale: float = DEFAULT_DELTA_XYZ_SCALE,
        precontact_offset: float = 0.025,
    ) -> None:
        if not np.isfinite(delta_xyz_scale) or delta_xyz_scale <= 0.0:
            raise ValueError("delta_xyz_scale must be finite and positive.")
        if not np.isfinite(precontact_offset) or precontact_offset < 0.0:
            raise ValueError("precontact_offset must be finite and nonnegative.")
        self.delta_xyz_scale = float(delta_xyz_scale)
        self.precontact_offset = float(precontact_offset)

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        tip = self._tip_position(observation)
        task = observation.get("task", {})
        if not isinstance(task, Mapping):
            raise ValueError("observation.task must be a mapping.")
        object_name = str(task.get("object_name", "push_object"))
        object_position = self._object_position(observation, object_name)
        goal = self._point3(task.get("goal_position"), "goal_position")
        phase = str(task.get("phase", "approach"))

        push_vector = goal - object_position
        push_norm = float(np.linalg.norm(push_vector))
        push_direction = (
            push_vector / push_norm
            if push_norm > 1e-12
            else np.zeros(3, dtype=np.float64)
        )
        if phase == "approach":
            target = object_position - self.precontact_offset * push_direction
        else:
            target = goal
        normalized_delta = np.clip(
            (target - tip) / self.delta_xyz_scale,
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
        if not isinstance(tip_pose, Mapping):
            raise ValueError("observation.robot_state.tip_pose is required.")
        return cls._point3(tip_pose.get("position"), "tip_pose.position")

    @classmethod
    def _object_position(
        cls,
        observation: Mapping[str, Any],
        object_name: str,
    ) -> np.ndarray:
        objects = observation.get("objects", {})
        if not isinstance(objects, Mapping):
            raise ValueError("observation.objects must be a mapping.")
        state = objects.get(object_name, {})
        if not isinstance(state, Mapping):
            raise ValueError(f"observation.objects.{object_name} is required.")
        pose = state.get("pose", {})
        if not isinstance(pose, Mapping):
            raise ValueError(f"{object_name}.pose is required.")
        return cls._point3(pose.get("position"), f"{object_name}.pose.position")

    @staticmethod
    def _point3(value: Any, label: str) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{label} must contain exactly three finite values.")
        return point
