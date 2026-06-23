from __future__ import annotations

from typing import Any, Mapping

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController
from soft_continuum_vlm.controllers.safety_projector import SafetyLimits, SafetyProjector
from soft_continuum_vlm.planners.deterministic_vlm_planner import DeterministicVLMPlanner


class VlmPlannerController:
    def __init__(
        self,
        *,
        planner: DeterministicVLMPlanner | None = None,
        controller: PccIkController | None = None,
        safety_projector: SafetyProjector | None = None,
    ) -> None:
        self.planner = planner or DeterministicVLMPlanner()
        self.controller = controller or PccIkController()
        self.safety_projector = safety_projector or SafetyProjector(
            SafetyLimits(
                max_abs_section_angle=0.3,
                max_gripper_rotation=1.0,
                max_contact_force=1.0,
                max_penetration=0.01,
            )
        )

    def act(
        self,
        *,
        language: str,
        observation: Mapping[str, Any],
        task_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        planner_output = self.planner.plan(language, observation, task_name)
        selected_subgoal = self._select_subgoal(planner_output)
        raw_action = self.controller.compute_action(
            target_state={"planner_output": planner_output, "selected_subgoal": selected_subgoal},
            robot_state=observation.get("robot_state", {}),
        )
        raw_action = self._shape_action(raw_action, selected_subgoal, planner_output)
        contact = observation.get("contact", {})
        contact_force = float(contact.get("max_force", 0.0)) if isinstance(contact, Mapping) else 0.0
        penetration = float(contact.get("max_penetration", 0.0)) if isinstance(contact, Mapping) else 0.0
        safe_action, safety_info = self.safety_projector.project(
            raw_action,
            contact_force=contact_force,
            penetration=penetration,
        )
        info = {
            "planner_output": planner_output,
            "selected_subgoal": selected_subgoal,
            "phase": selected_subgoal["phase"],
            "safety_info": safety_info,
            "raw_action": raw_action,
            "safe_action": safe_action,
        }
        return safe_action, info

    @staticmethod
    def _select_subgoal(planner_output: Mapping[str, Any]) -> dict[str, Any]:
        subgoals = planner_output.get("subgoals", [])
        if isinstance(subgoals, list) and subgoals:
            return dict(subgoals[0])
        return {"phase": "approach", "target": planner_output.get("target_object", "target_object")}

    @staticmethod
    def _shape_action(
        action: dict[str, Any],
        selected_subgoal: Mapping[str, Any],
        planner_output: Mapping[str, Any],
    ) -> dict[str, Any]:
        phase = str(selected_subgoal.get("phase", "approach"))
        action = dict(action)
        if phase in {"approach", "avoid", "grasp"}:
            action["section_angles"] = [0.15] * 6
        if phase == "grasp" or planner_output.get("grasp_mode") == "gentle":
            action["grip_command"] = 1.0
        if planner_output.get("language_constraints", {}).get("requires_rotation"):
            action["grasper_rotation"] = 0.8
        return action
