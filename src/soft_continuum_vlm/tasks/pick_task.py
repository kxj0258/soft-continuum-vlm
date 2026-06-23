from __future__ import annotations

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
