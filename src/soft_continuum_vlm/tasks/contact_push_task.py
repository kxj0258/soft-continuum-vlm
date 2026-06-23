from __future__ import annotations

from soft_continuum_vlm.tasks.base_task import BaseTask, TaskSpec


class ContactPushTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            TaskSpec(
                name="contact_push",
                language="Use safe contact to push the object to the marked region.",
                target_object="push_object",
                success={"requires_region_reached": True, "max_contact_force": 1.0},
            )
        )
