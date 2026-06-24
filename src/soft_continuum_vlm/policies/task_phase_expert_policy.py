from __future__ import annotations

from typing import Any, Mapping

from soft_continuum_vlm.controllers.task_phase_expert import TaskPhaseExpert


class TaskPhaseExpertPolicy:
    baseline_name = "task_phase_expert"

    def __init__(self, expert: TaskPhaseExpert | None = None) -> None:
        self.expert = expert or TaskPhaseExpert()

    def reset(self, task_name: str, language: str | None = None) -> None:
        self.expert.reset(task_name, language)

    def act(self, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.expert.act(observation)
