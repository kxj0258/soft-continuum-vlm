from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController
from soft_continuum_vlm.controllers.safety_projector import SafetyProjector


ACTION_KEYS = ("section_angles", "grip_command", "grasper_rotation")


class ScriptedExpert:
    """Minimal deterministic phase expert for the virtual pick_red_object task."""

    def __init__(
        self,
        task_name: str = "pick_red_object",
        *,
        pcc_controller: Any | None = None,
        controller: Any | None = None,
        safety_projector: Any | None = None,
        calibration_path: str | None = None,
        target_offset: Sequence[float] = (0.0, 0.02, 0.0),
        position_tolerance: float = 0.006,
        phase_timeout: int = 40,
        close_steps: int = 1,
        hold_steps: int = 1,
        hold_mode: str = "anchor",
        close_loop_hold: bool = True,
        close_hold_position_gain_scale: float = 0.35,
        max_post_close_drift: float = 0.02,
        overshoot_guard: bool = True,
        done_hold_mode: str = "closed_loop",
        done_hold_position_gain_scale: float = 0.20,
        done_overshoot_guard: bool = True,
        done_overshoot_margin: float = 0.003,
        done_max_section_step_norm: float = 0.03,
        done_continue_control: bool = True,
    ) -> None:
        self.task_name = task_name
        self.pcc_controller = pcc_controller or controller
        self._uses_fallback_controller = False
        if self.pcc_controller is None:
            self.pcc_controller = PccIkController(calibration_path=calibration_path) if calibration_path else PccIkController()
            self._uses_fallback_controller = calibration_path is None
        self.safety_projector: SafetyProjector | None = safety_projector
        self.target_offset = self._as_vector(target_offset, length=3)
        self.position_tolerance = float(position_tolerance)
        self.phase_timeout = int(phase_timeout)
        self.close_steps = int(close_steps)
        self.hold_steps = int(hold_steps)
        if hold_mode not in {"anchor", "current"}:
            raise ValueError("hold_mode must be 'anchor' or 'current'.")
        self.hold_mode = hold_mode
        self.close_loop_hold = bool(close_loop_hold)
        self.close_hold_position_gain_scale = float(close_hold_position_gain_scale)
        self.max_post_close_drift = float(max_post_close_drift)
        self.overshoot_guard = bool(overshoot_guard)
        if done_hold_mode not in {"closed_loop", "open_loop"}:
            raise ValueError("done_hold_mode must be 'closed_loop' or 'open_loop'.")
        self.done_hold_mode = done_hold_mode
        self.done_hold_position_gain_scale = float(done_hold_position_gain_scale)
        self.done_overshoot_guard = bool(done_overshoot_guard)
        self.done_overshoot_margin = float(done_overshoot_margin)
        self.done_max_section_step_norm = float(done_max_section_step_norm)
        self.done_continue_control = bool(done_continue_control)
        self.reset(task_name=task_name)

    def reset(self, task_name: str | None = None, language: str | None = None) -> None:
        if task_name is not None:
            self.task_name = task_name
        self.language = language
        self.phase = "approach_virtual_object"
        self.phase_step = 0
        self.total_step = 0
        self._virtual_red_object_position: list[float] | None = None
        self._virtual_red_object_source: str | None = None
        self._grasp_anchor_position: list[float] | None = None
        self._grasp_anchor_section_angles: list[float] | None = None
        self._grasp_anchor_grasper_rotation: float | None = None
        self._best_distance_to_target: float | None = None
        self._best_tip_position: list[float] | None = None
        self._best_step: int | None = None
        self._last_stable_action: dict[str, Any] | None = None
        self._last_stable_distance: float | None = None

    def act(self, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        robot_state = self._robot_state(observation)
        current_tip = self._current_tip_position(robot_state)
        current_section_angles = self._section_angles(robot_state)
        current_grasper_rotation = float(robot_state.get("grasper_rotation", 0.0))
        red_object_position, target_source = self._red_object_position(observation, current_tip)
        target_distance = float(np.linalg.norm(np.asarray(red_object_position) - np.asarray(current_tip)))
        phase_transition = self._advance_phase_if_needed(
            target_distance,
            current_tip,
            current_section_angles,
            current_grasper_rotation,
        )
        target_state = self._target_state(current_tip, red_object_position, current_grasper_rotation)
        post_close_drift = (
            float(target_distance - self._best_distance_to_target)
            if self.phase in {"close_gripper", "hold_closed", "done"} and self._best_distance_to_target is not None
            else None
        )
        drift_warning = post_close_drift is not None and post_close_drift >= self.max_post_close_drift
        overshoot_guard_triggered = (
            self.overshoot_guard
            and self.phase in {"close_gripper", "hold_closed"}
            and self._best_distance_to_target is not None
            and target_distance > self._best_distance_to_target + self.max_post_close_drift
        )
        done_overshoot_guard_triggered = False
        current_y_minus_target_y: float | None = None
        done_closed_loop_action: dict[str, Any] | None = None

        if self.phase == "approach_virtual_object":
            raw_action, ik_info = self.pcc_controller.compute_action_with_info(target_state, robot_state)
        elif self.phase in {"close_gripper", "hold_closed"} and overshoot_guard_triggered:
            raw_action = {
                "section_angles": [float(value) for value in current_section_angles],
                "grip_command": 1.0,
                "grasper_rotation": float(current_grasper_rotation),
            }
            ik_info = {
                "source": "scripted_expert",
                "status": "overshoot_guard_hold_current",
                "notes": ["Post-close drift exceeded max_post_close_drift; holding current section angles."],
            }
        elif self.phase in {"close_gripper", "hold_closed"} and self.close_loop_hold:
            pcc_action, ik_info = self.pcc_controller.compute_action_with_info(target_state, robot_state)
            raw_action = self._blend_section_action(
                pcc_action,
                current_section_angles,
                current_grasper_rotation,
                target_state,
            )
        elif self.phase == "done" and self.done_continue_control and self.done_hold_mode == "closed_loop":
            pcc_action, ik_info = self.pcc_controller.compute_action_with_info(target_state, robot_state)
            raw_action, done_hold_info = self._done_closed_loop_action(
                pcc_action,
                current_section_angles,
                current_tip,
                red_object_position,
                current_grasper_rotation,
                target_state,
            )
            done_overshoot_guard_triggered = bool(done_hold_info["done_overshoot_guard_triggered"])
            current_y_minus_target_y = done_hold_info["current_y_minus_target_y"]
            done_closed_loop_action = raw_action
        else:
            raw_action = self._last_stable_action or self._hold_action(
                current_section_angles,
                current_grasper_rotation,
                grip_command=1.0,
            )
            ik_info = {
                "source": "scripted_expert",
                "status": f"{self.phase}_hold_anchor",
                "notes": ["PCC IK is bypassed outside approach to avoid close/hold overrun."],
            }
        raw_action = self._complete_action(raw_action, current_section_angles, target_state, current_grasper_rotation)
        safe_action, safety_info = self._project_action(raw_action, observation, robot_state, current_section_angles)
        safe_action = self._complete_action(safe_action, current_section_angles, target_state, current_grasper_rotation)
        if self._is_stable_distance(target_distance):
            self._last_stable_action = dict(safe_action)
            self._last_stable_distance = float(target_distance)

        info = {
            "source": "scripted_expert",
            "task_name": self.task_name,
            "phase": self.phase,
            "phase_step": self.phase_step,
            "total_step": self.total_step,
            "target_source": target_source,
            "red_object_position": red_object_position,
            "target_state": self._jsonable(target_state),
            "target_distance": target_distance,
            "phase_transition": phase_transition,
            "ik_info": self._jsonable(ik_info),
            "raw_action": self._jsonable(raw_action),
            "safe_action": self._jsonable(safe_action),
            "safety": self._jsonable(safety_info),
            "best_distance_to_target": self._best_distance_to_target,
            "best_tip_position": self._best_tip_position,
            "best_step": self._best_step,
            "grasp_anchor_position": self._grasp_anchor_position,
            "grasp_anchor_section_angles": self._grasp_anchor_section_angles,
            "grasp_anchor_grasper_rotation": self._grasp_anchor_grasper_rotation,
            "hold_mode": self.hold_mode,
            "close_loop_hold": self.close_loop_hold,
            "close_hold_position_gain_scale": self.close_hold_position_gain_scale,
            "current_distance_to_target": target_distance,
            "post_close_drift": post_close_drift,
            "drift_warning": bool(drift_warning),
            "overshoot_guard_triggered": bool(overshoot_guard_triggered),
            "done_hold_mode": self.done_hold_mode,
            "done_continue_control": self.done_continue_control,
            "done_hold_position_gain_scale": self.done_hold_position_gain_scale,
            "done_max_section_step_norm": self.done_max_section_step_norm,
            "done_overshoot_guard": self.done_overshoot_guard,
            "done_overshoot_margin": self.done_overshoot_margin,
            "done_overshoot_guard_triggered": bool(done_overshoot_guard_triggered),
            "current_y_minus_target_y": current_y_minus_target_y,
            "done_closed_loop_action": self._jsonable(done_closed_loop_action),
            "last_stable_distance": self._last_stable_distance,
            "last_stable_action": self._last_stable_action,
            "notes": self._notes(),
        }
        self.total_step += 1
        self.phase_step += 1
        return safe_action, self._jsonable(info)

    def _advance_phase_if_needed(
        self,
        target_distance: float,
        current_tip: list[float],
        current_section_angles: list[float],
        current_grasper_rotation: float,
    ) -> dict[str, Any]:
        from_phase = self.phase
        reason: str | None = None
        self._update_best_distance(target_distance, current_tip)
        if self.phase == "approach_virtual_object":
            if target_distance < self.position_tolerance:
                self._record_grasp_anchor(current_tip, current_section_angles, current_grasper_rotation)
                self.phase = "close_gripper"
                reason = "distance_tolerance"
            elif self.phase_step >= self.phase_timeout:
                self._record_grasp_anchor(current_tip, current_section_angles, current_grasper_rotation)
                self.phase = "close_gripper"
                reason = "phase_timeout_anchor"
        elif self.phase == "close_gripper" and self.phase_step >= self.close_steps:
            self.phase = "hold_closed"
            reason = "close_steps"
        elif self.phase == "hold_closed" and self.phase_step >= self.hold_steps:
            self.phase = "done"
            reason = "hold_steps"

        changed = self.phase != from_phase
        if changed:
            self.phase_step = 0
        return {
            "changed": changed,
            "from": from_phase if changed else None,
            "to": self.phase if changed else None,
            "reason": reason,
        }

    def _target_state(
        self,
        current_tip: list[float],
        red_object_position: list[float],
        current_grasper_rotation: float,
    ) -> dict[str, Any]:
        if self.phase == "approach_virtual_object":
            return {
                "target_tip_position": red_object_position,
                "grip_command": 0.0,
                "grasper_rotation": current_grasper_rotation,
            }
        if self.phase == "close_gripper":
            return {
                "target_tip_position": red_object_position,
                "grip_command": 1.0,
                "grasper_rotation": current_grasper_rotation,
            }
        if self.phase == "hold_closed":
            return {
                "target_tip_position": red_object_position,
                "grip_command": 1.0,
                "grasper_rotation": current_grasper_rotation,
            }
        return {
            "target_tip_position": red_object_position,
            "grip_command": 1.0,
            "grasper_rotation": current_grasper_rotation,
        }

    def _update_best_distance(self, target_distance: float, current_tip: list[float]) -> None:
        if self._best_distance_to_target is None or target_distance < self._best_distance_to_target:
            self._best_distance_to_target = float(target_distance)
            self._best_tip_position = list(current_tip)
            self._best_step = int(self.total_step)

    def _is_stable_distance(self, target_distance: float) -> bool:
        if self._best_distance_to_target is None:
            return False
        return target_distance <= self._best_distance_to_target + self.max_post_close_drift

    def _red_object_position(
        self,
        observation: Mapping[str, Any],
        current_tip: list[float],
    ) -> tuple[list[float], str]:
        object_position = self._object_position(observation.get("objects"), "red_object")
        if object_position is not None:
            return object_position, "objects.red_object"
        virtual_position = self._object_position(observation.get("virtual_objects"), "red_object")
        if virtual_position is not None:
            return virtual_position, "virtual_objects.red_object"
        if self._virtual_red_object_position is None:
            self._virtual_red_object_position = (
                np.asarray(current_tip, dtype=np.float64) + np.asarray(self.target_offset, dtype=np.float64)
            ).tolist()
            self._virtual_red_object_source = "initial_tip_plus_offset"
        return list(self._virtual_red_object_position), str(self._virtual_red_object_source)

    def _record_grasp_anchor(
        self,
        current_tip: list[float],
        current_section_angles: list[float],
        current_grasper_rotation: float,
    ) -> None:
        self._grasp_anchor_position = list(current_tip)
        self._grasp_anchor_section_angles = list(current_section_angles)
        self._grasp_anchor_grasper_rotation = float(current_grasper_rotation)

    def _hold_action(
        self,
        current_section_angles: list[float],
        current_grasper_rotation: float,
        *,
        grip_command: float,
    ) -> dict[str, Any]:
        if self.hold_mode == "anchor":
            section_angles = self._grasp_anchor_section_angles or current_section_angles
            grasper_rotation = (
                self._grasp_anchor_grasper_rotation
                if self._grasp_anchor_grasper_rotation is not None
                else current_grasper_rotation
            )
        else:
            section_angles = current_section_angles
            grasper_rotation = current_grasper_rotation
        return {
            "section_angles": [float(value) for value in section_angles],
            "grip_command": float(grip_command),
            "grasper_rotation": float(grasper_rotation),
        }

    def _blend_section_action(
        self,
        raw_action: Mapping[str, Any],
        current_section_angles: list[float],
        current_grasper_rotation: float,
        target_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        complete_raw = self._complete_action(raw_action, current_section_angles, target_state, current_grasper_rotation)
        raw_angles = np.asarray(complete_raw["section_angles"], dtype=np.float64)
        current = np.asarray(current_section_angles, dtype=np.float64)
        if current.size != raw_angles.size:
            current = np.zeros_like(raw_angles)
        scale = self.close_hold_position_gain_scale
        blended = current + scale * (raw_angles - current)
        return {
            "section_angles": [float(value) for value in blended.tolist()],
            "grip_command": 1.0,
            "grasper_rotation": float(complete_raw.get("grasper_rotation", current_grasper_rotation)),
        }

    def _done_closed_loop_action(
        self,
        raw_action: Mapping[str, Any],
        current_section_angles: list[float],
        current_tip_position: list[float],
        red_object_position: list[float],
        current_grasper_rotation: float,
        target_state: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        complete_raw = self._complete_action(raw_action, current_section_angles, target_state, current_grasper_rotation)
        raw_angles = np.asarray(complete_raw["section_angles"], dtype=np.float64)
        current = np.asarray(current_section_angles, dtype=np.float64)
        if current.size != raw_angles.size:
            current = np.zeros_like(raw_angles)

        blended = current + self.done_hold_position_gain_scale * (raw_angles - current)
        delta = blended - current
        delta_norm = float(np.linalg.norm(delta))
        max_step_norm = float(self.done_max_section_step_norm)
        if delta_norm > max_step_norm > 0.0:
            delta = delta / delta_norm * max_step_norm
            blended = current + delta

        current_y_minus_target_y = float(current_tip_position[1] - red_object_position[1])
        guard_triggered = bool(
            self.done_overshoot_guard and current_y_minus_target_y > float(self.done_overshoot_margin)
        )
        if guard_triggered:
            blended = np.minimum(blended, current)

        action = {
            "section_angles": [float(value) for value in blended.tolist()],
            "grip_command": 1.0,
            "grasper_rotation": float(current_grasper_rotation),
        }
        return action, {
            "done_overshoot_guard_triggered": guard_triggered,
            "current_y_minus_target_y": current_y_minus_target_y,
        }

    @staticmethod
    def _object_position(objects: Any, object_id: str) -> list[float] | None:
        if not isinstance(objects, Mapping):
            return None
        item = objects.get(object_id)
        if not isinstance(item, Mapping):
            return None
        if item.get("available", True) is False:
            return None
        pose = item.get("pose")
        if not isinstance(pose, Mapping) or "position" not in pose:
            return None
        return ScriptedExpert._as_vector(pose["position"], length=3)

    @staticmethod
    def _robot_state(observation: Mapping[str, Any]) -> dict[str, Any]:
        robot_state = observation.get("robot_state")
        if isinstance(robot_state, Mapping):
            return dict(robot_state)
        return {"proprioception": observation.get("proprioception", [])}

    @staticmethod
    def _current_tip_position(robot_state: Mapping[str, Any]) -> list[float]:
        tip_pose = robot_state.get("tip_pose")
        if isinstance(tip_pose, Mapping) and "position" in tip_pose:
            return ScriptedExpert._as_vector(tip_pose["position"], length=3)
        return [0.0, 0.0, 0.0]

    @staticmethod
    def _section_angles(robot_state: Mapping[str, Any]) -> list[float]:
        values = robot_state.get("section_angles", [])
        if isinstance(values, str) or not isinstance(values, Sequence):
            return []
        return [float(value) for value in values]

    def _project_action(
        self,
        raw_action: Mapping[str, Any],
        observation: Mapping[str, Any],
        robot_state: Mapping[str, Any],
        current_section_angles: list[float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.safety_projector is None:
            return dict(raw_action), {"enabled": False}
        contact = observation.get("contact", {})
        contact_force = float(contact.get("max_force", 0.0)) if isinstance(contact, Mapping) else 0.0
        penetration = float(contact.get("max_penetration", 0.0)) if isinstance(contact, Mapping) else 0.0
        safe_action, safety_info = self.safety_projector.project(
            raw_action,
            contact_force=contact_force,
            penetration=penetration,
            current_robot_state={**dict(robot_state), "section_angles": current_section_angles},
        )
        return dict(safe_action), {"enabled": True, **dict(safety_info)}

    @staticmethod
    def _complete_action(
        action: Mapping[str, Any],
        current_section_angles: list[float],
        target_state: Mapping[str, Any],
        current_grasper_rotation: float,
    ) -> dict[str, Any]:
        section_angles = action.get("section_angles", current_section_angles)
        length = len(current_section_angles) if current_section_angles else len(section_angles)
        return {
            "section_angles": ScriptedExpert._as_vector(section_angles, length=length),
            "grip_command": float(action.get("grip_command", target_state.get("grip_command", 0.0))),
            "grasper_rotation": float(
                action.get("grasper_rotation", target_state.get("grasper_rotation", current_grasper_rotation))
            ),
        }

    def _notes(self) -> list[str]:
        notes = [
            "This is a minimal virtual pick_red_object expert; no real object registry or grasp physics is assumed."
        ]
        if self._uses_fallback_controller:
            notes.append("No calibration_path was provided; using PccIkController fallback Jacobian.")
        return notes

    @staticmethod
    def _as_vector(value: Any, *, length: int) -> list[float]:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size >= length:
            return [float(item) for item in array[:length].tolist()]
        padded = np.zeros(length, dtype=np.float64)
        padded[: array.size] = array
        return [float(item) for item in padded.tolist()]

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): ScriptedExpert._jsonable(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return [ScriptedExpert._jsonable(item) for item in value.tolist()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [ScriptedExpert._jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
