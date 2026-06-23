from __future__ import annotations

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
