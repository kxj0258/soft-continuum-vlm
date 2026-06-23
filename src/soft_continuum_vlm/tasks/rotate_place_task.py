from __future__ import annotations

from typing import Any, Mapping

from soft_continuum_vlm.tasks.base_task import BaseTask, TaskSpec


class RotatePlaceTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            TaskSpec(
                name="rotate_and_place",
                language="Rotate the grasped object and place it at the target pose.",
                target_object="grasped_object",
                success={"requires_pose_match": True, "max_pose_error": 0.02},
            )
        )

    def evaluate(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        target = self.target_object(observation)
        pose = target.get("pose", {})
        target_pose = target.get("target_pose", {})
        if not isinstance(pose, Mapping):
            pose = {}
        if not isinstance(target_pose, Mapping):
            target_pose = {}
        position_error = self.position_error(pose.get("position"), target_pose.get("position"))
        orientation_error = self.orientation_error(pose.get("orientation"), target_pose.get("orientation"))
        max_pose_error = float(self.spec.success.get("max_pose_error", 0.02))
        max_orientation_error = float(self.spec.success.get("max_orientation_error", 0.1))
        pose_match = position_error <= max_pose_error and orientation_error <= max_orientation_error
        return {
            "success": pose_match,
            "metrics": {
                "position_error": position_error,
                "orientation_error": orientation_error,
                "max_pose_error": max_pose_error,
                "max_orientation_error": max_orientation_error,
                "pose_match": pose_match,
            },
        }
