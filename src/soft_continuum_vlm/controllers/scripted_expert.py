from __future__ import annotations

from typing import Any, Mapping

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController
from soft_continuum_vlm.controllers.safety_projector import SafetyProjector


class ScriptedExpert:
    """Deterministic expert shell for scripted demonstration collection."""

    def __init__(
        self,
        controller: PccIkController | None = None,
        safety_projector: SafetyProjector | None = None,
    ) -> None:
        self.controller = controller or PccIkController()
        self.safety_projector = safety_projector

    def act(self, observation: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        robot_state = dict(observation.get("robot_state", {})) if isinstance(observation.get("robot_state"), Mapping) else {}
        if not robot_state:
            robot_state = {"proprioception": observation.get("proprioception", [])}
        action = self.controller.compute_action(target_state={}, robot_state=robot_state)
        info: dict[str, Any] = {
            "source": "scripted_pcc_ik",
            "deferred_note": (
                "Expected input: scene state, task spec, and robot state. Expected "
                "output: safe expert continuum action. Integration path: add PCC IK "
                "waypoint logic and task heuristics in Milestone 3."
            ),
        }
        if self.safety_projector is None:
            return action, info
        contact = observation.get("contact", {})
        contact_force = float(contact.get("max_force", 0.0)) if isinstance(contact, Mapping) else 0.0
        penetration = float(contact.get("max_penetration", 0.0)) if isinstance(contact, Mapping) else 0.0
        safe_action, safety_info = self.safety_projector.project(
            action,
            contact_force=contact_force,
            penetration=penetration,
        )
        info["safety"] = safety_info
        return safe_action, info
