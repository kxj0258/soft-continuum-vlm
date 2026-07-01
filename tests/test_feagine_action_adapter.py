from __future__ import annotations

import math

import numpy as np
import pytest

from soft_continuum_vlm.controllers.feagine_action_adapter import (
    FeagineActionAdapter,
)
from soft_continuum_vlm.controllers.ik import IkResult, IkSolver


class _RecordingSolver(IkSolver):
    def __init__(self, *, succeed: bool = True) -> None:
        self.succeed = succeed
        self.targets: list[list[float]] = []

    def solve(
        self,
        target_tip_position,
        current_section_angles,
        *,
        current_tip_position=None,
    ) -> IkResult:
        target = [float(value) for value in target_tip_position]
        self.targets.append(target)
        if not self.succeed:
            return IkResult.failure(
                current_section_angles,
                "unreachable",
                1.0,
                achieved_tip_position=current_tip_position,
            )
        return IkResult(
            success=True,
            converged=True,
            section_angles=[0.2] * len(current_section_angles),
            achieved_tip_position=target,
            position_error_norm=0.0,
            iterations=1,
            status="converged",
        )


def _observation() -> dict[str, object]:
    return {
        "robot_state": {
            "tip_pose": {"position": [0.1, 0.2, 0.3]},
            "section_angles": [0.0] * 6,
            "grip_command": 0.0,
            "grasper_rotation": 0.25,
        }
    }


def test_adapter_scales_xyz_and_maps_closed_gripper_to_runtime_command() -> None:
    solver = _RecordingSolver()
    adapter = FeagineActionAdapter(ik_solver=solver)

    conversion = adapter.convert_with_info([1.0, -0.5, 0.0, 1.0], _observation())

    np.testing.assert_allclose(conversion.target_tip_position, [0.11, 0.195, 0.3])
    assert conversion.command.section_angles == [0.2] * 6
    assert conversion.command.gripper_open_close == pytest.approx(1.0)
    assert conversion.command.to_runtime_action()["grip_command"] == pytest.approx(1.0)
    assert conversion.current_gripper_open_close == pytest.approx(0.0)


def test_adapter_maps_open_gripper_and_holds_rotation_without_task_context() -> None:
    adapter = FeagineActionAdapter(ik_solver=_RecordingSolver())

    command = adapter.convert([0.0, 0.0, 0.0, -1.0], _observation())

    assert command.gripper_open_close == pytest.approx(0.0)
    assert command.grasper_rotation == pytest.approx(0.25)


def test_adapter_zero_xyz_with_default_solver_holds_current_section_angles() -> None:
    adapter = FeagineActionAdapter()

    command = adapter.convert([0.0, 0.0, 0.0, 0.0], _observation())

    assert command.section_angles == [0.0] * 6


def test_adapter_computes_approach_rotation_from_target_direction() -> None:
    adapter = FeagineActionAdapter(ik_solver=_RecordingSolver())

    command = adapter.convert(
        [0.0, 0.0, 0.0, 0.0],
        _observation(),
        task_context={
            "phase": "approach",
            "orientation_target_position": [0.1, 1.2, 0.3],
        },
    )

    assert command.grasper_rotation == pytest.approx(math.pi / 2.0)


def test_adapter_holds_current_angles_after_all_ik_attempts_fail() -> None:
    adapter = FeagineActionAdapter(ik_solver=_RecordingSolver(succeed=False))

    conversion = adapter.convert_with_info([1.0, 0.0, 0.0, 0.0], _observation())

    assert conversion.command.section_angles == [0.0] * 6
    assert conversion.ik_result.status == "fallback_hold"
    assert conversion.ik_result.attempts == 3


def test_adapter_rejects_observation_without_required_robot_state() -> None:
    adapter = FeagineActionAdapter(ik_solver=_RecordingSolver())

    with pytest.raises(ValueError, match="tip_pose.position"):
        adapter.convert([0.0, 0.0, 0.0, 0.0], {"robot_state": {}})
