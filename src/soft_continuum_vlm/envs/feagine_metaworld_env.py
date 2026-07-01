from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from soft_continuum_vlm.controllers.feagine_action_adapter import FeagineActionAdapter
from soft_continuum_vlm.envs.action_space import FeagineActionSpace
from soft_continuum_vlm.tasks.feagine_metaworld_task import FeagineMetaWorldTask


class FeagineMetaWorldEnv:
    """Expose a 4D task interface over an existing low-level Feagine backend."""

    def __init__(
        self,
        backend: Any,
        task: FeagineMetaWorldTask,
        *,
        action_adapter: FeagineActionAdapter | None = None,
    ) -> None:
        self.backend = backend
        self.task = task
        self.action_adapter = action_adapter or FeagineActionAdapter()
        self.action_space = FeagineActionSpace()
        self._observation: dict[str, Any] = {}
        self._ik_attempts = 0
        self._ik_successes = 0
        self._distance_history: list[float] = []

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> dict[str, Any]:
        raw_observation = self.backend.reset(seed=seed, **kwargs)
        self.task.reset_task(seed=seed, observation=raw_observation)
        self._ik_attempts = 0
        self._ik_successes = 0
        self._distance_history = []
        self._observation = self._with_task_info(raw_observation)
        return self.get_observation()

    def step(
        self,
        action: Sequence[float],
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        conversion = self.action_adapter.convert_with_info(
            action,
            self._observation,
            task_context=self.task.get_task_context(self._observation),
        )
        runtime_action = conversion.command.to_runtime_action()
        raw_observation, _backend_reward, backend_done, backend_info = self.backend.step(
            runtime_action
        )
        self._observation = self._with_task_info(raw_observation)
        evaluation = self.task.evaluate(self._observation)

        self._ik_attempts += 1
        self._ik_successes += int(conversion.ik_result.success)
        distance = evaluation.metrics.get("tip_goal_distance")
        if distance is not None:
            self._distance_history.append(float(distance))
        success = bool(evaluation.success)
        done = success or bool(backend_done)
        info = {
            "task_name": self.task.name,
            "success": success,
            "backend_done": bool(backend_done),
            "task_metrics": dict(evaluation.metrics),
            "task_info": self.task.get_task_info(),
            "ik": asdict(conversion.ik_result),
            "ik_success_rate": float(self._ik_successes / self._ik_attempts),
            "tip_goal_distance_history": list(self._distance_history),
            "low_level_command": conversion.command.as_dict(),
            "runtime_action": runtime_action,
            "backend_info": dict(backend_info),
        }
        return self.get_observation(), float(evaluation.reward), done, info

    def render(self) -> Any:
        return self.backend.render()

    def close(self) -> None:
        self.backend.close()

    def get_observation(self) -> dict[str, Any]:
        return dict(self._observation)

    def get_contact_info(self) -> Mapping[str, Any]:
        contact = self._observation.get("contact", {})
        return dict(contact) if isinstance(contact, Mapping) else {}

    def get_robot_state(self) -> Mapping[str, Any]:
        robot_state = self._observation.get("robot_state", {})
        return dict(robot_state) if isinstance(robot_state, Mapping) else {}

    def _with_task_info(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(observation)
        result["language"] = self.task.language
        result["task"] = self.task.get_task_info()
        return result
