from __future__ import annotations

from typing import Sequence

import numpy as np

from soft_continuum_vlm.controllers.ik import MujocoFkDifferentialIkSolver


class _LinearTipProbe:
    def __init__(self) -> None:
        self.calls: list[list[float]] = []

    def probe_tip_position_for_section_angles(
        self,
        section_angles: Sequence[float],
    ) -> list[float]:
        angles = np.asarray(section_angles, dtype=np.float64)
        self.calls.append(angles.astype(float).tolist())
        return [float(angles[0]), float(angles[2]), 0.3]


def test_mujoco_fk_ik_solver_uses_probe_jacobian_to_reduce_error() -> None:
    probe = _LinearTipProbe()
    solver = MujocoFkDifferentialIkSolver(probe)

    result = solver.solve(
        [0.2, 0.0, 0.3],
        [0.0] * 6,
        current_tip_position=[0.0, 0.0, 0.3],
    )

    assert result.success
    assert result.status == "step_applied"
    assert result.position_error_norm < 0.2
    assert result.section_angles[0] > 0.0
    assert len(probe.calls) >= 13
