from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController
from soft_continuum_vlm.controllers.safety_projector import SafetyLimits, SafetyProjector
from soft_continuum_vlm.tasks.contact_push_task import ContactPushTask
from soft_continuum_vlm.tasks.obstacle_avoid_pick_task import ObstacleAvoidPickTask
from soft_continuum_vlm.tasks.pick_task import PickTask
from soft_continuum_vlm.tasks.rotate_place_task import RotatePlaceTask


@dataclass
class ExpertPhaseState:
    phase: str
    step_in_phase: int
    completed_phases: list[str] = field(default_factory=list)


TASK_PHASES = {
    "pick_red_object": ["approach_above_target", "approach_target", "close_gripper", "lift", "done"],
    "obstacle_avoid_pick": [
        "move_to_pre_avoid_waypoint",
        "arc_around_obstacle",
        "approach_target",
        "close_gripper",
        "lift",
        "done",
    ],
    "contact_push": ["approach_push_object", "make_safe_contact", "push_toward_region", "retract", "done"],
    "rotate_and_place": [
        "approach_object",
        "close_gripper",
        "rotate_grasper",
        "move_to_target_pose",
        "release",
        "done",
    ],
}

TASKS = {
    "pick_red_object": PickTask,
    "obstacle_avoid_pick": ObstacleAvoidPickTask,
    "contact_push": ContactPushTask,
    "rotate_and_place": RotatePlaceTask,
}

TARGET_OBJECTS = {
    "pick_red_object": "red_object",
    "obstacle_avoid_pick": "target_object",
    "contact_push": "push_object",
    "rotate_and_place": "grasped_object",
}


class TaskPhaseExpert:
    def __init__(
        self,
        *,
        controller: PccIkController | None = None,
        safety_projector: SafetyProjector | None = None,
        waypoint_tolerance: float = 0.05,
        phase_timeout: int = 12,
    ) -> None:
        self.controller = controller or PccIkController()
        self.safety_projector = safety_projector or SafetyProjector(
            SafetyLimits(
                max_abs_section_angle=0.8,
                max_gripper_rotation=1.0,
                max_contact_force=1.0,
                max_penetration=0.01,
            )
        )
        self.waypoint_tolerance = waypoint_tolerance
        self.phase_timeout = phase_timeout
        self.task_name = "pick_red_object"
        self.language = ""
        self.state = ExpertPhaseState(phase=TASK_PHASES[self.task_name][0], step_in_phase=0)
        self._previous_action: dict[str, Any] | None = None

    def reset(self, task_name: str, language: str | None = None) -> None:
        if task_name not in TASK_PHASES:
            raise ValueError(f"Unsupported task_name: {task_name}")
        self.task_name = task_name
        self.language = language or ""
        self.state = ExpertPhaseState(phase=TASK_PHASES[task_name][0], step_in_phase=0)
        self._previous_action = None

    def act(self, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        robot_state = observation.get("robot_state", {})
        if not isinstance(robot_state, Mapping):
            robot_state = {"proprioception": observation.get("proprioception", [])}
        target_state, target_distance = self._target_state(observation)
        raw_action = self.controller.compute_action(target_state, robot_state)
        contact = observation.get("contact", {})
        contact_force = float(contact.get("max_force", 0.0)) if isinstance(contact, Mapping) else 0.0
        penetration = float(contact.get("max_penetration", 0.0)) if isinstance(contact, Mapping) else 0.0
        safe_action, safety_info = self.safety_projector.project(
            raw_action,
            contact_force=contact_force,
            penetration=penetration,
            previous_action=self._previous_action,
            current_robot_state=robot_state,
            safety_mode="hold_current",
        )
        transition = self._maybe_advance_phase(observation, target_distance, target_state)
        self._previous_action = dict(safe_action)
        info = {
            "source": "task_phase_expert",
            "task_name": self.task_name,
            "phase": self.state.phase if not transition["advanced"] else transition["from"],
            "target_state": target_state,
            "raw_action": raw_action,
            "safe_action": safe_action,
            "safety": safety_info,
            "phase_transition": transition,
            "target_distance": target_distance,
        }
        return safe_action, info

    def _target_state(self, observation: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        phase = self.state.phase
        current_position = self._tip_position(observation)
        target_position = self._phase_target_position(observation, phase)
        target_state: dict[str, Any] = {
            "target_tip_position": target_position,
            "phase": phase,
        }
        if phase in {"close_gripper", "lift", "rotate_grasper", "move_to_target_pose"}:
            target_state["grip_command"] = 1.0
        if phase in {"release", "retract"}:
            target_state["grip_command"] = 0.0
        if phase == "rotate_grasper":
            target_state["grasper_rotation"] = 1.0
        return target_state, float(np.linalg.norm(np.asarray(target_position) - np.asarray(current_position)))

    def _phase_target_position(self, observation: Mapping[str, Any], phase: str) -> list[float]:
        target = self._object_position(observation, TARGET_OBJECTS[self.task_name])
        if phase in {"approach_above_target", "move_to_pre_avoid_waypoint", "approach_object"}:
            return [target[0], target[1], target[2] + 0.12]
        if phase == "arc_around_obstacle":
            obstacle = self._object_position(observation, "obstacle")
            return [max(target[0] - 0.08, 0.0), obstacle[1] + 0.12, target[2] + 0.10]
        if phase in {"approach_target", "approach_push_object", "make_safe_contact"}:
            return [target[0], target[1], target[2] + 0.02]
        if phase == "push_toward_region":
            push_object = self._object(observation, "push_object")
            region = push_object.get("target_region", {}) if isinstance(push_object, Mapping) else {}
            center = region.get("center", [target[0] + 0.15, target[1], target[2]])
            return [float(center[0]), float(center[1]), float(center[2])]
        if phase in {"lift", "retract"}:
            return [target[0], target[1], target[2] + 0.18]
        if phase == "move_to_target_pose":
            obj = self._object(observation, TARGET_OBJECTS[self.task_name])
            target_pose = obj.get("target_pose", {}) if isinstance(obj, Mapping) else {}
            return [float(value) for value in target_pose.get("position", [target[0], target[1], target[2] + 0.04])[:3]]
        return self._tip_position(observation)

    def _maybe_advance_phase(
        self,
        observation: Mapping[str, Any],
        target_distance: float,
        target_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        phases = TASK_PHASES[self.task_name]
        current_phase = self.state.phase
        task_result = TASKS[self.task_name]().evaluate(observation)
        reached = target_distance <= self.waypoint_tolerance
        timed_out = self.state.step_in_phase + 1 >= self.phase_timeout
        command_reached = self._command_reached(observation, target_state)
        success = bool(task_result.get("success", False))
        should_advance = current_phase != "done" and (success or reached or timed_out or command_reached)
        reason = "none"
        if success:
            reason = "task_success"
        elif reached:
            reason = "target_reached"
        elif command_reached:
            reason = "command_reached"
        elif timed_out:
            reason = "phase_timeout"
        if should_advance:
            current_index = phases.index(current_phase)
            next_phase = phases[min(current_index + 1, len(phases) - 1)]
            self.state.completed_phases.append(current_phase)
            self.state.phase = next_phase
            self.state.step_in_phase = 0
            return {"advanced": True, "from": current_phase, "to": next_phase, "reason": reason}
        self.state.step_in_phase += 1
        return {"advanced": False, "from": current_phase, "to": current_phase, "reason": reason}

    @staticmethod
    def _command_reached(observation: Mapping[str, Any], target_state: Mapping[str, Any]) -> bool:
        robot_state = observation.get("robot_state", {})
        if not isinstance(robot_state, Mapping):
            return False
        checks = []
        if "grip_command" in target_state:
            checks.append(abs(float(robot_state.get("grip_command", 0.0)) - float(target_state["grip_command"])) < 0.1)
        if "grasper_rotation" in target_state:
            checks.append(
                abs(float(robot_state.get("grasper_rotation", 0.0)) - float(target_state["grasper_rotation"])) < 0.1
            )
        return bool(checks and all(checks))

    @staticmethod
    def _tip_position(observation: Mapping[str, Any]) -> list[float]:
        robot_state = observation.get("robot_state", {})
        if isinstance(robot_state, Mapping):
            tip_pose = robot_state.get("tip_pose", {})
            if isinstance(tip_pose, Mapping) and "position" in tip_pose:
                return [float(value) for value in tip_pose["position"][:3]]
        return [0.0, 0.0, 0.0]

    def _object_position(self, observation: Mapping[str, Any], object_id: str) -> list[float]:
        obj = self._object(observation, object_id)
        pose = obj.get("pose", {}) if isinstance(obj, Mapping) else {}
        position = pose.get("position", [0.45, 0.0, 0.04]) if isinstance(pose, Mapping) else [0.45, 0.0, 0.04]
        return [float(value) for value in position[:3]]

    @staticmethod
    def _object(observation: Mapping[str, Any], object_id: str) -> Mapping[str, Any]:
        objects = observation.get("objects", {})
        if not isinstance(objects, Mapping):
            return {}
        obj = objects.get(object_id, {})
        return obj if isinstance(obj, Mapping) else {}
