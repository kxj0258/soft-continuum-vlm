from __future__ import annotations

from typing import Any, Mapping

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

    def evaluate(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        target = self.target_object(observation)
        pose = target.get("pose", {})
        target_region = target.get("target_region", {})
        if not isinstance(pose, Mapping):
            pose = {}
        if not isinstance(target_region, Mapping):
            target_region = {}
        object_position = self.vector3(pose.get("position"))
        region_center = self.vector3(target_region.get("center"))
        region_radius = float(target_region.get("radius", 0.0))
        region_distance = self.position_error(object_position, region_center)
        region_reached = region_distance <= region_radius
        contact = observation.get("contact", {})
        max_contact_force = float(contact.get("max_force", 0.0)) if isinstance(contact, Mapping) else 0.0
        force_limit = float(self.spec.success.get("max_contact_force", 1.0))
        contact_safe = max_contact_force <= force_limit
        return {
            "success": region_reached and contact_safe,
            "metrics": {
                "region_reached": region_reached,
                "region_distance": region_distance,
                "region_radius": region_radius,
                "max_contact_force": max_contact_force,
                "force_limit": force_limit,
                "contact_safe": contact_safe,
            },
        }
