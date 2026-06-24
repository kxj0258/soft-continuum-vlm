from __future__ import annotations

import pytest

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkConfig, PccIkController


def robot_state() -> dict[str, object]:
    return {
        "tip_pose": {"position": [0.0, 0.0, 0.0]},
        "section_angles": [0.0] * 6,
        "grip_command": 0.2,
        "grasper_rotation": -0.1,
    }


def test_pcc_ik_controller_outputs_nonzero_section_angles_for_positive_x_target() -> None:
    controller = PccIkController(PccIkConfig(section_count=3, max_step_norm=0.25))

    action = controller.compute_action(
        {"target_tip_position": [0.05, 0.0, 0.0], "phase": "approach"},
        robot_state(),
    )

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert max(abs(value) for value in action["section_angles"]) > 0.0
    assert action["grip_command"] == pytest.approx(0.2)
    assert action["grasper_rotation"] == pytest.approx(-0.1)


def test_pcc_ik_controller_uses_explicit_grip_and_rotation_targets() -> None:
    controller = PccIkController(PccIkConfig(section_count=3))

    action = controller.compute_action(
        {
            "target_tip_position": [0.0, 0.0, 0.0],
            "grip_command": 1.0,
            "grasper_rotation": 0.75,
            "phase": "grasp",
        },
        robot_state(),
    )

    assert action["section_angles"] == [0.0] * 6
    assert action["grip_command"] == pytest.approx(1.0)
    assert action["grasper_rotation"] == pytest.approx(0.75)


def test_pcc_ik_controller_limits_step_norm_and_angle_bounds() -> None:
    controller = PccIkController(PccIkConfig(section_count=3, max_step_norm=0.05, max_abs_section_angle=0.2))

    action = controller.compute_action(
        {"target_tip_position": [1.0, 1.0, 0.0], "phase": "approach"},
        robot_state(),
    )

    assert sum(value * value for value in action["section_angles"]) ** 0.5 <= 0.05 + 1e-9
    assert all(abs(value) <= 0.2 for value in action["section_angles"])
