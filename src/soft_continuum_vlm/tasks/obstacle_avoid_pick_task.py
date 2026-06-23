from __future__ import annotations

from typing import Any, Mapping

from soft_continuum_vlm.tasks.base_task import BaseTask, TaskSpec


class ObstacleAvoidPickTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            TaskSpec(
                name="obstacle_avoid_pick",
                language="Reach around the obstacle and pick the target object.",
                target_object="target_object",
                success={"requires_grasp": True, "max_obstacle_contact_force": 0.5},
            )
        )

    def evaluate(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        target = self.target_object(observation)
        target_grasped = bool(target.get("grasped", False))
        obstacle_contact_force = self.contact_force_for(observation, "obstacle")
        max_obstacle_contact_force = float(self.spec.success.get("max_obstacle_contact_force", 0.5))
        obstacle_contact_safe = obstacle_contact_force <= max_obstacle_contact_force
        success = target_grasped and obstacle_contact_safe
        return {
            "success": success,
            "metrics": {
                "target_grasped": target_grasped,
                "obstacle_contact_force": obstacle_contact_force,
                "max_obstacle_contact_force": max_obstacle_contact_force,
                "obstacle_contact_safe": obstacle_contact_safe,
            },
        }
