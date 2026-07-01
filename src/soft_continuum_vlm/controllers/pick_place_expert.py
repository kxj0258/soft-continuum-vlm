from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.envs.action_space import DEFAULT_DELTA_XYZ_SCALE


class FeaginePickPlaceExpert:
    """Deterministic staged expert that emits only normalized 4D actions."""

    CLOSED_PHASES = {"close_gripper", "lift", "transport", "align_place"}

    def __init__(self, *, delta_xyz_scale: float = DEFAULT_DELTA_XYZ_SCALE) -> None:
        if not np.isfinite(delta_xyz_scale) or delta_xyz_scale <= 0.0:
            raise ValueError("delta_xyz_scale must be finite and positive.")
        self.delta_xyz_scale = float(delta_xyz_scale)

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        tip = self._tip_position(observation)
        task = observation.get("task", {})
        if not isinstance(task, Mapping):
            raise ValueError("observation.task must be a mapping.")
        phase = str(task.get("phase", "approach"))
        object_name = str(task.get("object_name", "pick_object"))
        object_position = self._object_position(observation, object_name)
        goal = self._point3(task.get("goal_position"), "goal_position")
        lift_height = float(task.get("lift_height", 0.06))
        pick_position = self._point3(
            task.get("pick_position", object_position),
            "pick_position",
        )

        target = tip.copy()
        if phase == "approach":
            target = object_position
        elif phase == "lift":
            target = np.asarray(
                [object_position[0], object_position[1], pick_position[2] + lift_height]
            )
        elif phase == "transport":
            target = goal + np.asarray([0.0, 0.0, lift_height])
        elif phase == "align_place":
            target = goal
        elif phase == "retract":
            target = tip + np.asarray([0.0, 0.0, lift_height])

        normalized_delta = np.clip(
            (target - tip) / self.delta_xyz_scale,
            -1.0,
            1.0,
        )
        gripper_control = 1.0 if phase in self.CLOSED_PHASES else -1.0
        return np.asarray([*normalized_delta, gripper_control], dtype=np.float32)

    @classmethod
    def _tip_position(cls, observation: Mapping[str, Any]) -> np.ndarray:
        state = observation.get("robot_state", {})
        if not isinstance(state, Mapping):
            raise ValueError("observation.robot_state must be a mapping.")
        pose = state.get("tip_pose", {})
        if not isinstance(pose, Mapping):
            raise ValueError("observation.robot_state.tip_pose is required.")
        return cls._point3(pose.get("position"), "tip_pose.position")

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
