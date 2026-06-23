from __future__ import annotations

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
