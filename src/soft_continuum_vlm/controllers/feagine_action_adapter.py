from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from soft_continuum_vlm.controllers.ik import (
    DifferentialIkSolver,
    IkResult,
    IkSolver,
    solve_with_retries,
)
from soft_continuum_vlm.envs.action_space import (
    DEFAULT_DELTA_XYZ_SCALE,
    scale_action,
)


@dataclass(frozen=True)
class FeagineActionAdapterConfig:
    delta_xyz_scale: float = DEFAULT_DELTA_XYZ_SCALE
    retry_scales: tuple[float, ...] = (1.0, 0.5, 0.25)
    max_abs_grasper_rotation: float = math.pi


@dataclass(frozen=True)
class FeagineLowLevelCommand:
    """Semantic low-level command with an explicit runtime alias boundary."""

    section_angles: list[float]
    grasper_rotation: float
    gripper_open_close: float

    def as_dict(self) -> dict[str, object]:
        return {
            "section_angles": list(self.section_angles),
            "grasper_rotation": float(self.grasper_rotation),
            "gripper_open_close": float(self.gripper_open_close),
        }

    def to_runtime_action(self) -> dict[str, object]:
        return {
            "section_angles": list(self.section_angles),
            "grasper_rotation": float(self.grasper_rotation),
            "grip_command": float(self.gripper_open_close),
        }


@dataclass(frozen=True)
class FeagineActionConversion:
    command: FeagineLowLevelCommand
    target_tip_position: list[float]
    ik_result: IkResult
    current_gripper_open_close: float


class LinearGripperMapper:
    """Map normalized high-level intent to Feagine's 0=open, 1=closed range."""

    def map(self, gripper_control: float) -> float:
        value = float(np.clip(float(gripper_control), -1.0, 1.0))
        return float((value + 1.0) / 2.0)


class GrasperOrientationController:
    """Deterministic task-phase rules for the hidden grasper rotation."""

    def __init__(self, *, max_abs_rotation: float = math.pi) -> None:
        if not np.isfinite(max_abs_rotation) or max_abs_rotation <= 0.0:
            raise ValueError("max_abs_rotation must be finite and positive.")
        self.max_abs_rotation = float(max_abs_rotation)

    def compute(
        self,
        current_tip_position: Sequence[float],
        current_grasper_rotation: float,
        task_context: Mapping[str, Any] | None = None,
    ) -> float:
        context = task_context or {}
        phase = str(context.get("phase", "")).lower()

        desired = self._explicit_rotation(context)
        if desired is None and phase in {"approach", "pregrasp"}:
            desired = self._target_direction(current_tip_position, context)
        if desired is None and phase in {"grasp", "align_grasper"}:
            desired = self._axis_rotation(context.get("object_principal_axis"))
            if desired is None:
                desired = self._target_direction(current_tip_position, context)
        if desired is None and phase in {"place", "align_place"}:
            desired = self._axis_rotation(context.get("place_principal_axis"))
            if desired is None:
                desired = self._axis_rotation(context.get("target_principal_axis"))
        if desired is None:
            desired = float(current_grasper_rotation)

        normalized = math.atan2(math.sin(desired), math.cos(desired))
        return float(np.clip(normalized, -self.max_abs_rotation, self.max_abs_rotation))

    @staticmethod
    def _explicit_rotation(context: Mapping[str, Any]) -> float | None:
        value = context.get("desired_grasper_rotation")
        if value is None:
            return None
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("desired_grasper_rotation must be finite.")
        return result

    @staticmethod
    def _axis_rotation(value: Any) -> float | None:
        if value is None:
            return None
        if np.isscalar(value):
            result = float(value)
            if not np.isfinite(result):
                raise ValueError("principal-axis rotation must be finite.")
            return result
        axis = np.asarray(value, dtype=np.float64).reshape(-1)
        if axis.size < 2 or not np.all(np.isfinite(axis[:2])):
            raise ValueError("principal axis must contain at least two finite values.")
        if float(np.linalg.norm(axis[:2])) <= 1e-12:
            return None
        return float(math.atan2(float(axis[1]), float(axis[0])))

    @staticmethod
    def _target_direction(
        current_tip_position: Sequence[float],
        context: Mapping[str, Any],
    ) -> float | None:
        raw_target = context.get(
            "orientation_target_position",
            context.get("target_object_position"),
        )
        if raw_target is None:
            return None
        current = _point3(current_tip_position, "current_tip_position")
        target = _point3(raw_target, "orientation target")
        direction = target[:2] - current[:2]
        if float(np.linalg.norm(direction)) <= 1e-12:
            return None
        return float(math.atan2(float(direction[1]), float(direction[0])))


class FeagineActionAdapter:
    """Convert one normalized 4D action into a bounded Feagine command."""

    def __init__(
        self,
        *,
        ik_solver: IkSolver | None = None,
        config: FeagineActionAdapterConfig | None = None,
        orientation_controller: GrasperOrientationController | None = None,
        gripper_mapper: LinearGripperMapper | None = None,
    ) -> None:
        self.config = config or FeagineActionAdapterConfig()
        self.ik_solver = ik_solver or DifferentialIkSolver()
        self.orientation_controller = orientation_controller or GrasperOrientationController(
            max_abs_rotation=self.config.max_abs_grasper_rotation
        )
        self.gripper_mapper = gripper_mapper or LinearGripperMapper()

    def convert(
        self,
        action_4d: Sequence[float] | np.ndarray,
        observation: Mapping[str, Any],
        *,
        task_context: Mapping[str, Any] | None = None,
    ) -> FeagineLowLevelCommand:
        return self.convert_with_info(
            action_4d,
            observation,
            task_context=task_context,
        ).command

    def convert_with_info(
        self,
        action_4d: Sequence[float] | np.ndarray,
        observation: Mapping[str, Any],
        *,
        task_context: Mapping[str, Any] | None = None,
    ) -> FeagineActionConversion:
        current_tip, current_angles, current_rotation, current_gripper = self._robot_state(
            observation
        )
        scaled = scale_action(
            action_4d,
            delta_xyz_scale=self.config.delta_xyz_scale,
        )
        target_tip = current_tip + scaled.delta_xyz.astype(np.float64)
        ik_result = solve_with_retries(
            self.ik_solver,
            target_tip,
            current_angles,
            current_tip_position=current_tip,
            retry_scales=self.config.retry_scales,
        )
        resolved_context = task_context
        if resolved_context is None:
            raw_context = observation.get("task", {})
            resolved_context = raw_context if isinstance(raw_context, Mapping) else {}
        rotation = self.orientation_controller.compute(
            current_tip,
            current_rotation,
            resolved_context,
        )
        command = FeagineLowLevelCommand(
            section_angles=list(ik_result.section_angles),
            grasper_rotation=rotation,
            gripper_open_close=self.gripper_mapper.map(scaled.gripper_control),
        )
        return FeagineActionConversion(
            command=command,
            target_tip_position=target_tip.astype(float).tolist(),
            ik_result=ik_result,
            current_gripper_open_close=current_gripper,
        )

    @staticmethod
    def _robot_state(
        observation: Mapping[str, Any],
    ) -> tuple[np.ndarray, list[float], float, float]:
        raw_state = observation.get("robot_state", observation)
        if not isinstance(raw_state, Mapping):
            raise ValueError("observation.robot_state must be a mapping.")
        tip_pose = raw_state.get("tip_pose", {})
        if not isinstance(tip_pose, Mapping) or "position" not in tip_pose:
            raise ValueError("observation.robot_state.tip_pose.position is required.")
        tip = _point3(tip_pose["position"], "tip_pose.position")
        raw_angles = raw_state.get("section_angles")
        if raw_angles is None or isinstance(raw_angles, (str, bytes)):
            raise ValueError("observation.robot_state.section_angles is required.")
        try:
            angles = np.asarray(raw_angles, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("section_angles must be a non-empty finite sequence.") from exc
        if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
            raise ValueError("section_angles must be a non-empty finite sequence.")
        rotation = float(raw_state.get("grasper_rotation", 0.0))
        if not np.isfinite(rotation):
            raise ValueError("grasper_rotation must be finite.")
        gripper = float(raw_state.get("grip_command", 0.0))
        if not np.isfinite(gripper):
            raise ValueError("grip_command must be finite.")
        return tip, angles.astype(float).tolist(), rotation, gripper


def _point3(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain exactly three finite values.")
    return array
