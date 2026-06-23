from __future__ import annotations

from typing import Any, Mapping

from soft_continuum_vlm.tasks.base_task import BaseTask, TaskSpec


class PickTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            TaskSpec(
                name="pick_red_object",
                language="Pick up the red object with contact-safe motion.",
                target_object="red_object",
                success={"requires_grasp": True, "requires_lift": True},
            )
        )

    def evaluate(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        target = self.target_object(observation)
        pose = self.target_pose(observation)
        position = self.vector3(pose.get("position"))
        target_grasped = bool(target.get("grasped", False))
        lift_height = position[2]
        min_lift_height = float(self.spec.success.get("min_lift_height", 0.1))
        target_lifted = lift_height >= min_lift_height
        success = target_grasped and target_lifted
        return {
            "success": success,
            "metrics": {
                "target_grasped": target_grasped,
                "target_lifted": target_lifted,
                "lift_height": lift_height,
                "min_lift_height": min_lift_height,
            },
        }
