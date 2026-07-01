from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FeagineTaskEvaluation:
    reward: float
    success: bool
    metrics: dict[str, float]


class FeagineMetaWorldTask(ABC):
    """Task-only contract independent from MuJoCo and action conversion."""

    name: str
    language: str

    @abstractmethod
    def reset_task(
        self,
        *,
        seed: int | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        """Reset deterministic task state and sample a goal when needed."""

    @abstractmethod
    def compute_reward(self, observation: Mapping[str, Any]) -> float:
        """Compute the task reward from the post-step observation."""

    @abstractmethod
    def compute_success(self, observation: Mapping[str, Any]) -> bool:
        """Return whether the task success condition is satisfied."""

    @abstractmethod
    def get_goal(self) -> list[float]:
        """Return the current goal in world coordinates."""

    @abstractmethod
    def get_object_state(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return task-relevant object state, or an empty mapping."""

    @abstractmethod
    def get_task_info(self) -> dict[str, Any]:
        """Return serializable task metadata for observations and logging."""

    def get_task_context(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return {"phase": "task", "goal_position": self.get_goal()}

    def update_task_state(self, observation: Mapping[str, Any]) -> None:
        del observation

    def compute_metrics(self, observation: Mapping[str, Any]) -> dict[str, float]:
        return {}

    def evaluate(self, observation: Mapping[str, Any]) -> FeagineTaskEvaluation:
        self.update_task_state(observation)
        return FeagineTaskEvaluation(
            reward=float(self.compute_reward(observation)),
            success=bool(self.compute_success(observation)),
            metrics=self.compute_metrics(observation),
        )

    @staticmethod
    def tip_position(observation: Mapping[str, Any]) -> np.ndarray:
        robot_state = observation.get("robot_state", {})
        if not isinstance(robot_state, Mapping):
            raise ValueError("observation.robot_state must be a mapping.")
        tip_pose = robot_state.get("tip_pose", {})
        if not isinstance(tip_pose, Mapping) or "position" not in tip_pose:
            raise ValueError("observation.robot_state.tip_pose.position is required.")
        position = np.asarray(tip_pose["position"], dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("tip_pose.position must contain exactly three finite values.")
        return position
