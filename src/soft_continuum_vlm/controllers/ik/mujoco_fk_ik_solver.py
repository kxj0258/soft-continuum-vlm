from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from soft_continuum_vlm.controllers.continuum_kinematics import ContinuumGeometry
from soft_continuum_vlm.controllers.ik.base_ik_solver import (
    IkResult,
    IkSolver,
    clip_section_angles,
    point3,
    section_angles,
)


class TipPositionProbe(Protocol):
    def probe_tip_position_for_section_angles(
        self,
        section_angles: Sequence[float],
    ) -> Sequence[float]:
        """Return the MuJoCo-forwarded tip position for a low-level command."""


@dataclass(frozen=True)
class MujocoFkIkConfig:
    position_tolerance: float = 1e-4
    max_step_norm: float = 0.08
    finite_difference_eps: float = 1e-4
    min_progress: float = 1e-12


class MujocoFkDifferentialIkSolver(IkSolver):
    """One-step differential IK using the backend's real MuJoCo FK response."""

    def __init__(
        self,
        probe: TipPositionProbe,
        config: MujocoFkIkConfig | None = None,
        *,
        geometry: ContinuumGeometry | None = None,
    ) -> None:
        self.probe = probe
        self.config = config or MujocoFkIkConfig()
        self.geometry = geometry or ContinuumGeometry()
        if self.config.finite_difference_eps <= 0.0:
            raise ValueError("finite_difference_eps must be positive.")

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
        current_tip = (
            point3(current_tip_position, "current_tip_position")
            if current_tip_position is not None
            else self._probe(current_angles)
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

        jacobian = self._numeric_jacobian(current_angles)
        matrix = jacobian @ jacobian.T + (self.geometry.damping**2) * np.eye(3)
        try:
            delta = jacobian.T @ np.linalg.solve(matrix, error)
        except np.linalg.LinAlgError:
            delta = jacobian.T @ np.linalg.pinv(matrix) @ error
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.config.max_step_norm > 0.0:
            delta *= self.config.max_step_norm / delta_norm

        next_angles = clip_section_angles(
            current_angles + delta,
            max_abs_section_angle=self.geometry.max_abs_section_angle,
        )
        achieved = self._probe(next_angles)
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

    def _numeric_jacobian(self, current_angles: np.ndarray) -> np.ndarray:
        eps = float(self.config.finite_difference_eps)
        jacobian = np.zeros((3, current_angles.size), dtype=np.float64)
        for index in range(current_angles.size):
            plus = current_angles.copy()
            minus = current_angles.copy()
            plus[index] += eps
            minus[index] -= eps
            plus = clip_section_angles(
                plus,
                max_abs_section_angle=self.geometry.max_abs_section_angle,
            )
            minus = clip_section_angles(
                minus,
                max_abs_section_angle=self.geometry.max_abs_section_angle,
            )
            jacobian[:, index] = (self._probe(plus) - self._probe(minus)) / (2.0 * eps)
        return jacobian

    def _probe(self, angles: Sequence[float]) -> np.ndarray:
        return point3(
            self.probe.probe_tip_position_for_section_angles(angles),
            "probe_tip_position",
        )
