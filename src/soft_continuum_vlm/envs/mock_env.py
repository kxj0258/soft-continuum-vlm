from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.data.schema import validate_action
from soft_continuum_vlm.envs.base_env import Action, BaseRobotEnv, Observation
from soft_continuum_vlm.tasks.contact_push_task import ContactPushTask
from soft_continuum_vlm.tasks.obstacle_avoid_pick_task import ObstacleAvoidPickTask
from soft_continuum_vlm.tasks.pick_task import PickTask
from soft_continuum_vlm.tasks.rotate_place_task import RotatePlaceTask


TASKS = {
    "pick_red_object": PickTask,
    "obstacle_avoid_pick": ObstacleAvoidPickTask,
    "contact_push": ContactPushTask,
    "rotate_and_place": RotatePlaceTask,
}


class MockContinuumEnv(BaseRobotEnv):
    """Deterministic headless environment for data and training smoke tests."""

    def __init__(self, *, task: str = "pick_red_object", section_count: int = 3, max_steps: int = 50) -> None:
        self.task_name = task
        self.section_count = section_count
        self.max_steps = max_steps
        self._task = TASKS[task]()
        self._rng = np.random.default_rng(0)
        self._step_count = 0
        self._closed = False
        self._language = self._task.language
        self._section_angles = np.zeros(2 * section_count, dtype=np.float32)
        self._grip_command = 0.0
        self._grasper_rotation = 0.0
        self._tip_position = np.asarray([0.1, 0.0, 0.1], dtype=np.float32)
        self._tip_orientation = [1.0, 0.0, 0.0, 0.0]
        self._objects: dict[str, dict[str, Any]] = {}
        self._contact: dict[str, Any] = {"max_force": 0.0, "max_penetration": 0.0, "contacts": []}
        self._observation = self._make_observation()

    def reset(
        self,
        language: str | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> Observation:
        if task is not None:
            self.task_name = task
            self._task = TASKS[task]()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._closed = False
        self._language = language or self._task.language
        self._section_angles = np.zeros(2 * self.section_count, dtype=np.float32)
        self._grip_command = 0.0
        self._grasper_rotation = 0.0
        self._tip_position = np.asarray([0.1, 0.0, 0.1], dtype=np.float32)
        self._contact = {"max_force": 0.0, "max_penetration": 0.0, "contacts": []}
        self._objects = self._initial_objects()
        self._observation = self._make_observation()
        return self.get_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        validate_action(action)
        self._section_angles = np.asarray(action["section_angles"], dtype=np.float32)
        self._grip_command = float(action["grip_command"])
        self._grasper_rotation = float(action["grasper_rotation"])
        self._advance_tip()
        self._advance_objects()
        self._step_count += 1
        self._observation = self._make_observation()
        result = self._task.evaluate(self._observation)
        done = bool(result["success"]) or self._step_count >= self.max_steps
        reward = 1.0 if result["success"] else -0.01
        info = {
            "task_name": self.task_name,
            "step_count": self._step_count,
            "success": bool(result["success"]),
            "metrics": result["metrics"],
        }
        return self.get_observation(), float(reward), done, info

    def render(self) -> Any:
        return self._observation["rgb"]

    def close(self) -> None:
        self._closed = True

    def get_observation(self) -> Observation:
        return dict(self._observation)

    def get_contact_info(self) -> Mapping[str, Any]:
        return dict(self._contact)

    def get_robot_state(self) -> Mapping[str, Any]:
        return dict(self._observation["robot_state"])

    def _initial_objects(self) -> dict[str, dict[str, Any]]:
        base = {
            "red_object": self._object([0.5, 0.0, 0.03]),
            "target_object": self._object([0.5, 0.0, 0.03]),
            "push_object": {
                **self._object([0.42, 0.0, 0.03]),
                "target_region": {"center": [0.6, 0.0, 0.03], "radius": 0.06},
            },
            "grasped_object": {
                **self._object([0.4, 0.1, 0.03]),
                "target_pose": {
                    "position": [0.405, 0.1, 0.03],
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                },
            },
        }
        return base

    @staticmethod
    def _object(position: list[float]) -> dict[str, Any]:
        return {
            "pose": {"position": list(position), "orientation": [1.0, 0.0, 0.0, 0.0]},
            "grasped": False,
        }

    def _advance_tip(self) -> None:
        bend_x = float(np.mean(self._section_angles[0::2])) if self._section_angles.size else 0.0
        bend_y = float(np.mean(self._section_angles[1::2])) if self._section_angles.size else 0.0
        self._tip_position[0] = min(0.7, max(0.0, self._tip_position[0] + 0.04 + 0.05 * bend_x))
        self._tip_position[1] = min(0.3, max(-0.3, self._tip_position[1] + 0.05 * bend_y))
        self._tip_position[2] = 0.12 + 0.04 * self._grip_command

    def _advance_objects(self) -> None:
        self._contact = {"max_force": 0.0, "max_penetration": 0.0, "contacts": []}
        target_id = self._task.spec.target_object
        target = self._objects.get(target_id)
        if target is None:
            return
        position = np.asarray(target["pose"]["position"], dtype=np.float32)
        distance = float(np.linalg.norm(self._tip_position[:2] - position[:2]))
        if self.task_name in {"pick_red_object", "obstacle_avoid_pick"}:
            if self._grip_command > 0.5 and distance < 0.18:
                target["grasped"] = True
            if target.get("grasped"):
                target["pose"]["position"] = self._tip_position.tolist()
        if self.task_name == "contact_push" and distance < 0.18:
            pushed = position.copy()
            pushed[0] = min(0.62, pushed[0] + 0.05)
            target["pose"]["position"] = pushed.tolist()
            self._contact = self._contact_between("finger_left", target_id, force=[0.6, 0.0, 0.0])
        if self.task_name == "rotate_and_place":
            target["pose"]["position"] = [0.405, 0.1, 0.03]
            target["pose"]["orientation"] = [1.0, 0.0, 0.0, 0.0] if abs(self._grasper_rotation) > 0.5 else [0.99, 0.1, 0.0, 0.0]

    @staticmethod
    def _contact_between(geom1: str, geom2: str, *, force: list[float]) -> dict[str, Any]:
        norm = float(np.linalg.norm(np.asarray(force, dtype=np.float32)))
        return {
            "max_force": norm,
            "max_penetration": 0.0,
            "contacts": [
                {
                    "geom1": geom1,
                    "geom2": geom2,
                    "position": [0.5, 0.0, 0.03],
                    "normal": [1.0, 0.0, 0.0],
                    "force": force,
                    "distance": 0.0,
                }
            ],
        }

    def _make_observation(self) -> Observation:
        robot_state = {
            "tip_pose": {
                "position": self._tip_position.astype(float).tolist(),
                "orientation": list(self._tip_orientation),
            },
            "section_angles": self._section_angles.astype(float).tolist(),
            "grip_command": float(self._grip_command),
            "grasper_rotation": float(self._grasper_rotation),
        }
        proprioception = np.asarray(
            [*robot_state["section_angles"], robot_state["grip_command"], robot_state["grasper_rotation"]],
            dtype=np.float32,
        )
        return {
            "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
            "depth": np.zeros((64, 64), dtype=np.float32),
            "language": self._language,
            "proprioception": proprioception,
            "robot_state": robot_state,
            "objects": self._objects,
            "contact": self._contact,
        }
