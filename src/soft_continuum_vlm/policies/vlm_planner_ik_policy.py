from __future__ import annotations

from typing import Any, Mapping

from soft_continuum_vlm.controllers.task_phase_expert import TaskPhaseExpert
from soft_continuum_vlm.planners.deterministic_vlm_planner import DeterministicVLMPlanner


class VlmPlannerIkPolicy:
    baseline_name = "vlm_planner_ik"

    def __init__(
        self,
        *,
        planner: DeterministicVLMPlanner | None = None,
        expert: TaskPhaseExpert | None = None,
    ) -> None:
        self.planner = planner or DeterministicVLMPlanner()
        self.expert = expert or TaskPhaseExpert()
        self.task_name = "pick_red_object"
        self.language = ""

    def reset(self, task_name: str, language: str | None = None) -> None:
        self.task_name = task_name
        self.language = language or ""
        self.expert.reset(task_name, language)

    def act(self, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        language = str(observation.get("language", self.language))
        planner_output = self.planner.plan(language, observation, self.task_name)
        selected_subgoal = self._selected_subgoal(planner_output)
        action, info = self.expert.act(observation)
        info = dict(info)
        info.update(
            {
                "source": "vlm_planner_ik_policy",
                "planner_output": planner_output,
                "selected_subgoal": selected_subgoal,
                "phase": info.get("phase", selected_subgoal.get("phase", "")),
            }
        )
        return action, info

    @staticmethod
    def _selected_subgoal(planner_output: Mapping[str, Any]) -> dict[str, Any]:
        subgoals = planner_output.get("subgoals", [])
        if isinstance(subgoals, list) and subgoals:
            return dict(subgoals[0])
        return {"phase": "approach", "target": planner_output.get("target_object", "target_object")}
