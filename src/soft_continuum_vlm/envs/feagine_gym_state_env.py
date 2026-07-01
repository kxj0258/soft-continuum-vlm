from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from soft_continuum_vlm.envs.feagine_metaworld_env import FeagineMetaWorldEnv
from soft_continuum_vlm.tasks.feagine_metaworld_task import FeagineMetaWorldTask


class FeagineGymStateEnv:
    """Small Gymnasium-style state wrapper over the 4D MetaWorld task env.

    This class intentionally avoids importing gymnasium so the research scaffold
    can expose the reset/step contract without adding an RL dependency yet.
    """

    def __init__(
        self,
        backend: Any,
        task: FeagineMetaWorldTask,
        *,
        metaworld_env: FeagineMetaWorldEnv | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        self.backend = backend
        self.task = task
        self.env = metaworld_env or FeagineMetaWorldEnv(backend, task)
        self.action_space = self.env.action_space
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._last_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        raw_observation = self.env.reset(seed=seed, **kwargs)
        self._step_count = 0
        metrics = self.task.compute_metrics(raw_observation)
        info = self._info(
            success=bool(self.task.compute_success(raw_observation)),
            backend_done=False,
            task_metrics=metrics,
            task_info=self.task.get_task_info(),
            backend_info={},
        )
        self._last_info = info
        return self._state_observation(raw_observation), dict(info)

    def step(
        self,
        action: Sequence[float],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        raw_observation, reward, done, info = self.env.step(action)
        self._step_count += 1
        success = bool(info.get("success", False))
        backend_done = bool(info.get("backend_done", False))
        terminated = bool(success or backend_done)
        truncated = bool(
            self.max_episode_steps is not None
            and self._step_count >= int(self.max_episode_steps)
            and not terminated
        )
        result_info = dict(info)
        result_info["step_count"] = self._step_count
        result_info["terminated"] = terminated
        result_info["truncated"] = truncated
        self._last_info = result_info
        return (
            self._state_observation(raw_observation),
            float(reward),
            terminated,
            truncated,
            result_info,
        )

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def get_observation(self) -> dict[str, np.ndarray]:
        return self._state_observation(self.env.get_observation())

    def get_raw_observation(self) -> dict[str, Any]:
        return self.env.get_observation()

    def get_info(self) -> dict[str, Any]:
        return dict(self._last_info)

    def _state_observation(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        achieved_goal = self._tip_position(observation).astype(np.float32, copy=False)
        desired_goal = np.asarray(self.task.get_goal(), dtype=np.float32)
        return {
            "observation": self._state_vector(observation),
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def _state_vector(self, observation: Mapping[str, Any]) -> np.ndarray:
        robot_state = observation.get("robot_state", {})
        if not isinstance(robot_state, Mapping):
            robot_state = {}
        tip = self._tip_position(observation)
        section_angles = self._finite_vector(
            robot_state.get("section_angles", []),
            label="robot_state.section_angles",
        )
        grip_command = self._finite_scalar(robot_state.get("grip_command", 0.0), "grip_command")
        grasper_rotation = self._finite_scalar(
            robot_state.get("grasper_rotation", 0.0),
            "grasper_rotation",
        )
        goal = np.asarray(self.task.get_goal(), dtype=np.float64)
        object_state = self._task_object_features(observation)
        contact = observation.get("contact", {})
        contact = contact if isinstance(contact, Mapping) else {}
        contact_features = np.asarray(
            [
                self._finite_scalar(contact.get("max_force", 0.0), "contact.max_force"),
                self._finite_scalar(contact.get("max_penetration", 0.0), "contact.max_penetration"),
                self._finite_scalar(contact.get("target_contact_count", 0.0), "contact.target_contact_count"),
                self._finite_scalar(contact.get("robot_contact_count", 0.0), "contact.robot_contact_count"),
            ],
            dtype=np.float64,
        )
        values = np.concatenate(
            [
                tip,
                goal,
                section_angles,
                np.asarray([grip_command, grasper_rotation], dtype=np.float64),
                object_state,
                contact_features,
            ]
        )
        return values.astype(np.float32, copy=False)

    def _task_object_features(self, observation: Mapping[str, Any]) -> np.ndarray:
        try:
            object_state = self.task.get_object_state(observation)
        except ValueError:
            return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        pose = object_state.get("pose", {}) if isinstance(object_state, Mapping) else {}
        position = (
            self._point3(pose.get("position"), "object.pose.position")
            if isinstance(pose, Mapping) and "position" in pose
            else np.zeros(3, dtype=np.float64)
        )
        grasped = 1.0 if bool(object_state.get("grasped", False)) else 0.0
        return np.asarray([position[0], position[1], position[2], grasped], dtype=np.float64)

    def _info(
        self,
        *,
        success: bool,
        backend_done: bool,
        task_metrics: Mapping[str, float],
        task_info: Mapping[str, Any],
        backend_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "task_name": self.task.name,
            "success": bool(success),
            "backend_done": bool(backend_done),
            "step_count": self._step_count,
            "task_metrics": dict(task_metrics),
            "task_info": dict(task_info),
            "backend_info": dict(backend_info),
            "terminated": bool(success or backend_done),
            "truncated": False,
        }

    @classmethod
    def _tip_position(cls, observation: Mapping[str, Any]) -> np.ndarray:
        robot_state = observation.get("robot_state", {})
        if not isinstance(robot_state, Mapping):
            raise ValueError("observation.robot_state must be a mapping.")
        tip_pose = robot_state.get("tip_pose", {})
        if not isinstance(tip_pose, Mapping):
            raise ValueError("observation.robot_state.tip_pose is required.")
        return cls._point3(tip_pose.get("position"), "tip_pose.position")

    @staticmethod
    def _finite_vector(value: Any, *, label: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain only finite values.")
        return array

    @staticmethod
    def _finite_scalar(value: Any, label: str) -> float:
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{label} must be finite.")
        return result

    @staticmethod
    def _point3(value: Any, label: str) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{label} must contain exactly three finite values.")
        return point
