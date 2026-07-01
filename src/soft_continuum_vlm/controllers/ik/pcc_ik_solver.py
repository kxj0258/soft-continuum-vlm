from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from soft_continuum_vlm.controllers.continuum_kinematics import (
    ContinuumGeometry,
    numeric_jacobian_tip_delta,
    section_angles_to_tip_delta,
)
from soft_continuum_vlm.controllers.ik.base_ik_solver import (
    IkResult,
    IkSolver,
    point3,
    section_angles,
)


@dataclass(frozen=True)
class PccIkConfig:
    max_iterations: int = 80
    position_tolerance: float = 1e-4
    max_step_norm: float = 0.12
    min_step_norm: float = 1e-10


class PccIkSolver(IkSolver):
    """Iterative global IK over the deterministic approximate PCC model."""

    def __init__(
        self,
        config: PccIkConfig | None = None,
        *,
        geometry: ContinuumGeometry | None = None,
    ) -> None:
        self.config = config or PccIkConfig()
        self.geometry = geometry or ContinuumGeometry()

    def solve(
        self,
        target_tip_position: Sequence[float],
        current_section_angles: Sequence[float],
        *,
        current_tip_position: Sequence[float] | None = None,
    ) -> IkResult:
        target = point3(target_tip_position, "target_tip_position")
        initial_angles = section_angles(
            current_section_angles,
            expected=2 * self.geometry.section_count,
            max_abs_section_angle=self.geometry.max_abs_section_angle,
        )
        current_tip = (
            point3(current_tip_position, "current_tip_position")
            if current_tip_position is not None
            else section_angles_to_tip_delta(initial_angles, self.geometry)
        )
        initial_model_tip = section_angles_to_tip_delta(initial_angles, self.geometry)
        angles = initial_angles.copy()

        for iteration in range(self.config.max_iterations + 1):
            predicted = current_tip + (
                section_angles_to_tip_delta(angles, self.geometry) - initial_model_tip
            )
            error = target - predicted
            error_norm = float(np.linalg.norm(error))
            if error_norm <= self.config.position_tolerance:
                return IkResult(
                    success=True,
                    converged=True,
                    section_angles=angles.astype(float).tolist(),
                    achieved_tip_position=predicted.astype(float).tolist(),
                    position_error_norm=error_norm,
                    iterations=iteration,
                    status="converged",
                )
            if iteration == self.config.max_iterations:
                break

            jacobian = numeric_jacobian_tip_delta(angles, self.geometry)
            matrix = jacobian @ jacobian.T + (self.geometry.damping**2) * np.eye(3)
            try:
                delta = jacobian.T @ np.linalg.solve(matrix, error)
            except np.linalg.LinAlgError:
                delta = jacobian.T @ np.linalg.pinv(matrix) @ error
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > self.config.max_step_norm > 0.0:
                delta *= self.config.max_step_norm / delta_norm
            candidate = np.clip(
                angles + delta,
                -self.geometry.max_abs_section_angle,
                self.geometry.max_abs_section_angle,
            )
            if float(np.linalg.norm(candidate - angles)) <= self.config.min_step_norm:
                return IkResult.failure(
                    initial_angles,
                    "no_progress",
                    float(np.linalg.norm(target - current_tip)),
                    achieved_tip_position=current_tip,
                    iterations=iteration,
                )
            angles = candidate

        return IkResult.failure(
            initial_angles,
            "max_iterations",
            float(np.linalg.norm(target - current_tip)),
            achieved_tip_position=current_tip,
            iterations=self.config.max_iterations,
        )
