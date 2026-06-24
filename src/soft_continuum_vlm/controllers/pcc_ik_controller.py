from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.controllers.calibrated_jacobian import JacobianCalibration


def zero_feagine_action(section_count: int) -> dict[str, object]:
    return {
        "section_angles": [0.0] * (2 * section_count),
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    }


@dataclass
class PccIkConfig:
    section_angle_length: int = 6
    max_abs_section_angle: float = 0.8
    max_step_norm: float = 0.15
    position_gain: float = 1.0
    damping: float = 1e-3
    active_columns_only: bool = True
    calibration_path: str | None = None
    section_count: int | None = None
    section_length: float = 0.10
    min_target_distance: float = 0.0

    def __post_init__(self) -> None:
        if self.section_count is not None:
            self.section_angle_length = 2 * int(self.section_count)
        self.section_angle_length = int(self.section_angle_length)


class PccIkController:
    """Local DLS controller backed by a calibrated section-angle Jacobian."""

    def __init__(
        self,
        config: PccIkConfig | None = None,
        *,
        jacobian: np.ndarray | None = None,
        calibration_path: str | Path | None = None,
        section_count: int | None = None,
    ) -> None:
        if config is None:
            config = PccIkConfig(section_count=section_count)
        elif section_count is not None:
            config.section_count = int(section_count)
            config.section_angle_length = 2 * int(section_count)
        self.config = config

        requested_calibration = calibration_path or config.calibration_path
        self.calibration: JacobianCalibration | None = None
        if requested_calibration is not None:
            self.calibration = JacobianCalibration.load_json(requested_calibration)
            self.jacobian = self.calibration.as_array()
            self.section_angle_length = int(self.calibration.section_angle_length)
            self.active_columns = list(self.calibration.active_columns)
            self.inactive_columns = list(self.calibration.inactive_columns)
            self.jacobian_source = "calibration"
        elif jacobian is not None:
            self.jacobian = self._validate_jacobian(jacobian)
            self.section_angle_length = int(self.jacobian.shape[1])
            self.active_columns = self._nonzero_columns(self.jacobian)
            self.inactive_columns = [
                index for index in range(self.section_angle_length) if index not in self.active_columns
            ]
            self.jacobian_source = "fallback"
        else:
            self.section_angle_length = int(config.section_angle_length)
            self.jacobian = self._fallback_jacobian(self.section_angle_length)
            self.active_columns = self._nonzero_columns(self.jacobian)
            self.inactive_columns = [
                index for index in range(self.section_angle_length) if index not in self.active_columns
            ]
            self.jacobian_source = "fallback"
        self.section_count = max(1, self.section_angle_length // 2)

    def compute_action(
        self,
        target_state: Mapping[str, Any],
        robot_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        action, _info = self.compute_action_with_info(target_state, robot_state)
        return action

    def compute_action_with_info(
        self,
        target_state: Mapping[str, Any],
        robot_state: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        notes: list[str] = []
        current_angles = self._current_section_angles(robot_state, notes)
        grip_command = float(target_state.get("grip_command", robot_state.get("grip_command", 0.0)))
        grasper_rotation = float(
            target_state.get("grasper_rotation", robot_state.get("grasper_rotation", 0.0))
        )
        target_position = target_state.get("target_tip_position")
        current_position = self._current_tip_position(robot_state)

        if target_position is None:
            action = self._action(current_angles, grip_command, grasper_rotation)
            info = self._info(
                status="no_target_hold_current",
                target_tip_position=None,
                current_tip_position=current_position,
                position_error=np.zeros(3, dtype=np.float64),
                dq=np.zeros(self.section_angle_length, dtype=np.float64),
                step_scaled=False,
                before=current_angles,
                after=current_angles,
                notes=notes,
            )
            return action, info

        if current_position is None:
            notes.append("robot_state.tip_pose.position is missing; holding current section_angles.")
            action = self._action(current_angles, grip_command, grasper_rotation)
            info = self._info(
                status="missing_tip_pose",
                target_tip_position=self._position_or_none(target_position),
                current_tip_position=None,
                position_error=np.zeros(3, dtype=np.float64),
                dq=np.zeros(self.section_angle_length, dtype=np.float64),
                step_scaled=False,
                before=current_angles,
                after=current_angles,
                notes=notes,
            )
            return action, info

        target = np.asarray(target_position, dtype=np.float64).reshape(-1)[:3]
        current = np.asarray(current_position, dtype=np.float64).reshape(-1)[:3]
        error = target - current
        dq, status, step_scaled = self._solve_dq(error)
        current_array = np.asarray(current_angles, dtype=np.float64)
        next_angles_array = np.clip(
            current_array + dq,
            -float(self.config.max_abs_section_angle),
            float(self.config.max_abs_section_angle),
        )
        next_angles = [float(value) for value in next_angles_array.tolist()]
        action = self._action(next_angles, grip_command, grasper_rotation)
        info = self._info(
            status=status,
            target_tip_position=target,
            current_tip_position=current,
            position_error=error,
            dq=dq,
            step_scaled=step_scaled,
            before=current_angles,
            after=next_angles,
            notes=notes,
        )
        return action, info

    def _solve_dq(self, error: np.ndarray) -> tuple[np.ndarray, str, bool]:
        use_columns = list(self.active_columns) if self.config.active_columns_only else list(range(self.section_angle_length))
        status = "ok"
        if not use_columns:
            use_columns = list(range(self.section_angle_length))
            status = "empty_active_columns"

        dq = np.zeros(self.section_angle_length, dtype=np.float64)
        if not use_columns:
            return dq, status, False

        jacobian_active = self.jacobian[:, use_columns]
        rhs = float(self.config.position_gain) * np.asarray(error, dtype=np.float64).reshape(3)
        matrix = jacobian_active @ jacobian_active.T + float(self.config.damping) * np.eye(3)
        try:
            solved = jacobian_active.T @ np.linalg.inv(matrix) @ rhs
        except np.linalg.LinAlgError:
            solved = jacobian_active.T @ np.linalg.pinv(matrix) @ rhs
        for local_index, column_index in enumerate(use_columns):
            dq[column_index] = float(solved[local_index])

        step_scaled = False
        dq_norm = float(np.linalg.norm(dq))
        max_step_norm = float(self.config.max_step_norm)
        if dq_norm > max_step_norm > 0.0:
            dq *= max_step_norm / dq_norm
            step_scaled = True
        return dq, status, step_scaled

    def _current_section_angles(self, robot_state: Mapping[str, Any], notes: list[str]) -> list[float]:
        raw = robot_state.get("section_angles")
        values = list(raw or [])
        if len(values) != self.section_angle_length:
            notes.append(
                f"section_angles length {len(values)} did not match expected {self.section_angle_length}; using zeros."
            )
            return [0.0] * self.section_angle_length
        return [float(value) for value in values]

    @staticmethod
    def _current_tip_position(robot_state: Mapping[str, Any]) -> list[float] | None:
        tip_pose = robot_state.get("tip_pose")
        if isinstance(tip_pose, Mapping) and "position" in tip_pose:
            return [float(value) for value in list(tip_pose["position"])[:3]]
        return None

    @staticmethod
    def _position_or_none(value: Any) -> list[float] | None:
        try:
            return [float(item) for item in list(value)[:3]]
        except TypeError:
            return None

    @staticmethod
    def _validate_jacobian(jacobian: np.ndarray) -> np.ndarray:
        array = np.asarray(jacobian, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != 3:
            raise ValueError(f"jacobian must have shape (3, N), got {array.shape}.")
        return array.copy()

    @staticmethod
    def _nonzero_columns(jacobian: np.ndarray) -> list[int]:
        norms = np.linalg.norm(jacobian, axis=0)
        return [int(index) for index, value in enumerate(norms) if float(value) > 0.0]

    @staticmethod
    def _fallback_jacobian(section_angle_length: int) -> np.ndarray:
        jacobian = np.zeros((3, section_angle_length), dtype=np.float64)
        for index, scale in zip([0, 2, 4], [0.25, 0.15, 0.05]):
            if index < section_angle_length:
                jacobian[1, index] = scale
                jacobian[2, index] = -0.003 * (scale / 0.25)
        return jacobian

    @staticmethod
    def _action(section_angles: list[float], grip_command: float, grasper_rotation: float) -> dict[str, Any]:
        return {
            "section_angles": [float(value) for value in section_angles],
            "grip_command": float(grip_command),
            "grasper_rotation": float(grasper_rotation),
        }

    def _info(
        self,
        *,
        status: str,
        target_tip_position: Any,
        current_tip_position: Any,
        position_error: np.ndarray,
        dq: np.ndarray,
        step_scaled: bool,
        before: list[float],
        after: list[float],
        notes: list[str],
    ) -> dict[str, Any]:
        error = np.asarray(position_error, dtype=np.float64).reshape(-1)[:3]
        return {
            "source": "pcc_ik_controller",
            "status": status,
            "jacobian_source": self.jacobian_source,
            "target_tip_position": self._to_optional_float_list(target_tip_position),
            "current_tip_position": self._to_optional_float_list(current_tip_position),
            "position_error": [float(value) for value in error.tolist()],
            "position_error_norm": float(np.linalg.norm(error)),
            "active_columns": [int(value) for value in self.active_columns],
            "inactive_columns": [int(value) for value in self.inactive_columns],
            "dq": [float(value) for value in np.asarray(dq, dtype=np.float64).reshape(-1).tolist()],
            "dq_norm": float(np.linalg.norm(np.asarray(dq, dtype=np.float64).reshape(-1))),
            "step_scaled": bool(step_scaled),
            "section_angles_before": [float(value) for value in before],
            "section_angles_after": [float(value) for value in after],
            "max_abs_section_angle": float(self.config.max_abs_section_angle),
            "notes": list(notes),
        }

    @staticmethod
    def _to_optional_float_list(value: Any) -> list[float] | None:
        if value is None:
            return None
        return [float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1)[:3].tolist()]
