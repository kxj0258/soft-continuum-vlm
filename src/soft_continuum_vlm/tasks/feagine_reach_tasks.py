from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from soft_continuum_vlm.tasks.feagine_metaworld_task import FeagineMetaWorldTask


class FeagineReachTask(FeagineMetaWorldTask):
    def __init__(
        self,
        *,
        name: str,
        language: str,
        goal_position: Sequence[float],
        success_threshold: float = 0.02,
    ) -> None:
        self.name = name
        self.language = language
        self.success_threshold = float(success_threshold)
        if not np.isfinite(self.success_threshold) or self.success_threshold <= 0.0:
            raise ValueError("success_threshold must be finite and positive.")
        self._goal = self._point3(goal_position, "goal_position")

    def reset_task(
        self,
        *,
        seed: int | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        del seed, observation

    def compute_reward(self, observation: Mapping[str, Any]) -> float:
        return -self.tip_goal_distance(observation)

    def compute_success(self, observation: Mapping[str, Any]) -> bool:
        return bool(self.tip_goal_distance(observation) < self.success_threshold)

    def get_goal(self) -> list[float]:
        return self._goal.astype(float).tolist()

    def get_object_state(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        del observation
        return {}

    def get_task_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language,
            "goal_position": self.get_goal(),
            "success_threshold": float(self.success_threshold),
            "task_family": "reach",
        }

    def get_task_context(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        del observation
        return {"phase": "reach", "goal_position": self.get_goal()}

    def compute_metrics(self, observation: Mapping[str, Any]) -> dict[str, float]:
        return {"tip_goal_distance": self.tip_goal_distance(observation)}

    def tip_goal_distance(self, observation: Mapping[str, Any]) -> float:
        return float(np.linalg.norm(self.tip_position(observation) - self._goal))

    @staticmethod
    def _point3(value: Sequence[float], label: str) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{label} must contain exactly three finite values.")
        return point


class FeagineReachLeftTask(FeagineReachTask):
    def __init__(
        self,
        *,
        goal_position: Sequence[float] = (-0.08, 0.0, 0.24),
        success_threshold: float = 0.02,
    ) -> None:
        super().__init__(
            name="feagine_reach_left",
            language="Move the Feagine tip to the left reach target.",
            goal_position=goal_position,
            success_threshold=success_threshold,
        )


class FeagineReachRightTask(FeagineReachTask):
    def __init__(
        self,
        *,
        goal_position: Sequence[float] = (0.08, 0.0, 0.24),
        success_threshold: float = 0.02,
    ) -> None:
        super().__init__(
            name="feagine_reach_right",
            language="Move the Feagine tip to the right reach target.",
            goal_position=goal_position,
            success_threshold=success_threshold,
        )


class FeagineReach3DTask(FeagineReachTask):
    def __init__(
        self,
        *,
        goal_low: Sequence[float] = (-0.08, -0.08, 0.20),
        goal_high: Sequence[float] = (0.08, 0.08, 0.28),
        success_threshold: float = 0.02,
    ) -> None:
        self.goal_low = self._point3(goal_low, "goal_low").astype(float).tolist()
        self.goal_high = self._point3(goal_high, "goal_high").astype(float).tolist()
        if np.any(np.asarray(self.goal_low) >= np.asarray(self.goal_high)):
            raise ValueError("goal_low must be strictly below goal_high.")
        super().__init__(
            name="feagine_reach_3d",
            language="Move the Feagine tip to the sampled three-dimensional target.",
            goal_position=(0.0, 0.0, 0.24),
            success_threshold=success_threshold,
        )

    def reset_task(
        self,
        *,
        seed: int | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        del observation
        rng = np.random.default_rng(seed)
        low = np.asarray(self.goal_low, dtype=np.float64)
        high = np.asarray(self.goal_high, dtype=np.float64)
        center = (low + high) / 2.0
        radii = (high - low) / 2.0
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = float(rng.random() ** (1.0 / 3.0))
        self._goal = center + radii * direction * radius

    def get_task_info(self) -> dict[str, Any]:
        info = super().get_task_info()
        info["goal_low"] = list(self.goal_low)
        info["goal_high"] = list(self.goal_high)
        info["sampling"] = "ellipsoid"
        return info


FEAGINE_REACH_TASKS = {
    "feagine_reach_left": FeagineReachLeftTask,
    "feagine_reach_right": FeagineReachRightTask,
    "feagine_reach_3d": FeagineReach3DTask,
}


def make_feagine_reach_task(name: str) -> FeagineReachTask:
    try:
        task_type = FEAGINE_REACH_TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Feagine reach task: {name}") from exc
    return task_type()
