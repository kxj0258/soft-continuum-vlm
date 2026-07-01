from __future__ import annotations

import numpy as np

from soft_continuum_vlm.controllers.ik import (
    DifferentialIkSolver,
    IkResult,
    IkSolver,
    PccIkSolver,
    solve_with_retries,
)
from soft_continuum_vlm.controllers.ik.base_ik_solver import section_angles


def test_pcc_ik_solver_reaches_small_target_and_respects_angle_limits() -> None:
    solver = PccIkSolver()
    current_angles = [0.0] * 6
    current_tip = [0.0, 0.0, 0.3]

    result = solver.solve(
        [0.02, 0.0, 0.295],
        current_angles,
        current_tip_position=current_tip,
    )

    assert result.success
    assert result.converged
    assert result.position_error_norm <= solver.config.position_tolerance
    assert all(abs(value) <= solver.geometry.max_abs_section_angle for value in result.section_angles[0::2])


def test_pcc_ik_solver_holds_current_angles_for_unreachable_target() -> None:
    solver = PccIkSolver()
    current = [0.1, 0.0, 0.1, 0.0, 0.1, 0.0]

    result = solver.solve(
        [10.0, 0.0, 10.0],
        current,
        current_tip_position=[0.0, 0.0, 0.3],
    )

    assert not result.success
    assert result.section_angles == current
    assert result.status in {"max_iterations", "no_progress"}


def test_differential_ik_solver_returns_one_continuous_bounded_step() -> None:
    solver = DifferentialIkSolver()
    current = np.zeros(6, dtype=np.float64)

    result = solver.solve(
        [0.005, 0.0, 0.3],
        current,
        current_tip_position=[0.0, 0.0, 0.3],
    )

    assert result.success
    assert result.iterations == 1
    assert np.linalg.norm(np.asarray(result.section_angles) - current) <= solver.config.max_step_norm + 1e-12
    assert result.position_error_norm < 0.005


def test_ik_accepts_direction_angles_larger_than_bend_limit() -> None:
    solver = DifferentialIkSolver()
    current = [0.0, np.pi / 2.0, 0.0, np.pi / 2.0, 0.0, np.pi / 2.0]

    result = solver.solve(
        [0.0, 0.0, 0.3],
        current,
        current_tip_position=[0.0, 0.0, 0.3],
    )

    assert result.success
    assert result.section_angles == current


def test_ik_accepts_real_feagine_bends_above_legacy_limit() -> None:
    solver = DifferentialIkSolver()
    current = [1.2, np.pi / 2.0, 0.8, -0.32, 0.4, np.pi]

    result = solver.solve(
        [0.0, 0.0, 0.3],
        current,
        current_tip_position=[0.0, 0.0, 0.3],
    )

    assert result.success
    np.testing.assert_allclose(result.section_angles, current)


def test_section_angle_validation_limits_only_bend_magnitudes() -> None:
    parsed = section_angles(
        [0.0, np.pi / 2.0, 0.2, -np.pi / 2.0, -0.2, np.pi],
        expected=6,
        max_abs_section_angle=0.8,
    )

    assert parsed[0::2].tolist() == [0.0, 0.2, -0.2]
    assert np.isclose(parsed[1], np.pi / 2.0)
    assert np.isclose(parsed[3], -np.pi / 2.0)
    assert np.isclose(parsed[5], np.pi)


class _DistanceLimitedSolver(IkSolver):
    def solve(
        self,
        target_tip_position,
        current_section_angles,
        *,
        current_tip_position=None,
    ) -> IkResult:
        distance = float(
            np.linalg.norm(
                np.asarray(target_tip_position, dtype=np.float64)
                - np.asarray(current_tip_position, dtype=np.float64)
            )
        )
        if distance > 0.3:
            return IkResult.failure(current_section_angles, "outside_local_region", distance)
        return IkResult(
            success=True,
            converged=True,
            section_angles=[0.2] * len(current_section_angles),
            achieved_tip_position=list(target_tip_position),
            position_error_norm=0.0,
            iterations=1,
            status="converged",
        )


def test_retry_reduces_target_delta_before_accepting_solution() -> None:
    result = solve_with_retries(
        _DistanceLimitedSolver(),
        [1.0, 0.0, 0.0],
        [0.0] * 6,
        current_tip_position=[0.0, 0.0, 0.0],
        retry_scales=(1.0, 0.5, 0.25),
    )

    assert result.success
    assert result.attempts == 3
    assert result.target_scale == 0.25


def test_retry_holds_current_angles_when_every_attempt_fails() -> None:
    current = [0.1] * 6
    result = solve_with_retries(
        _DistanceLimitedSolver(),
        [2.0, 0.0, 0.0],
        current,
        current_tip_position=[0.0, 0.0, 0.0],
        retry_scales=(1.0, 0.75),
    )

    assert not result.success
    assert result.status == "fallback_hold"
    assert result.section_angles == current
    assert result.attempts == 2
