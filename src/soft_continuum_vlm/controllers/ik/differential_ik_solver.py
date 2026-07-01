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
class DifferentialIkConfig:
    position_tolerance: float = 1e-4
    max_step_norm: float = 0.08
    min_progress: float = 1e-12


class DifferentialIkSolver(IkSolver):
    """One damped-Jacobian step for continuous small Cartesian commands."""

    def __init__(
        self,
        config: DifferentialIkConfig | None = None,
        *,
        geometry: ContinuumGeometry | None = None,
    ) -> None:
        self.config = config or DifferentialIkConfig()
        self.geometry = geometry or ContinuumGeometry()

    def solve(
        self,
        target_tip_position: Sequence[float],
        current_section_angles: Sequence[float],
        *,
        current_tip_position: Sequence[float] | None = None,
    ) -> IkResult:
        target = point3(target_tip_position, "target_tip_position")
        current_angles = section_angles(
            current_section_angles,
            expected=2 * self.geometry.section_count,
            max_abs_section_angle=self.geometry.max_abs_section_angle,
        )
        model_tip = section_angles_to_tip_delta(current_angles, self.geometry)
        current_tip = (
            point3(current_tip_position, "current_tip_position")
            if current_tip_position is not None
            else model_tip
        )
        error = target - current_tip
        initial_error_norm = float(np.linalg.norm(error))
        if initial_error_norm <= self.config.position_tolerance:
            return IkResult(
                success=True,
                converged=True,
                section_angles=current_angles.astype(float).tolist(),
                achieved_tip_position=current_tip.astype(float).tolist(),
                position_error_norm=initial_error_norm,
                iterations=0,
                status="converged",
            )

        jacobian = numeric_jacobian_tip_delta(current_angles, self.geometry)
        matrix = jacobian @ jacobian.T + (self.geometry.damping**2) * np.eye(3)
        try:
            delta = jacobian.T @ np.linalg.solve(matrix, error)
        except np.linalg.LinAlgError:
            delta = jacobian.T @ np.linalg.pinv(matrix) @ error
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.config.max_step_norm > 0.0:
            delta *= self.config.max_step_norm / delta_norm
        next_angles = np.clip(
            current_angles + delta,
            -self.geometry.max_abs_section_angle,
            self.geometry.max_abs_section_angle,
        )
        achieved = current_tip + (
            section_angles_to_tip_delta(next_angles, self.geometry) - model_tip
        )
        residual = float(np.linalg.norm(target - achieved))
        if not np.all(np.isfinite(next_angles)) or residual >= initial_error_norm - self.config.min_progress:
            return IkResult.failure(
                current_angles,
                "no_progress",
                initial_error_norm,
                achieved_tip_position=current_tip,
                iterations=1,
            )
        converged = residual <= self.config.position_tolerance
        return IkResult(
            success=True,
            converged=converged,
            section_angles=next_angles.astype(float).tolist(),
            achieved_tip_position=achieved.astype(float).tolist(),
            position_error_norm=residual,
            iterations=1,
            status="converged" if converged else "step_applied",
        )
