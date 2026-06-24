from __future__ import annotations

import numpy as np
import pytest

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkConfig, PccIkController


def _robot_state(section_angles: list[float] | None = None) -> dict[str, object]:
    return {
        "tip_pose": {"position": [0.0, 0.0, 0.0]},
        "section_angles": section_angles if section_angles is not None else [0.0] * 6,
        "grip_command": 0.2,
        "grasper_rotation": -0.1,
    }


def _test_jacobian() -> np.ndarray:
    jacobian = np.zeros((3, 6), dtype=np.float64)
    jacobian[1, 0] = 0.25
    jacobian[1, 2] = 0.15
    jacobian[1, 4] = 0.05
    return jacobian


def test_pcc_ik_no_target_holds_current() -> None:
    current = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]
    controller = PccIkController(jacobian=_test_jacobian())

    action, info = controller.compute_action_with_info({}, _robot_state(current))

    assert action["section_angles"] == pytest.approx(current)
    assert info["status"] == "no_target_hold_current"


def test_pcc_ik_target_y_changes_active_columns() -> None:
    controller = PccIkController(jacobian=_test_jacobian())

    action, info = controller.compute_action_with_info(
        {"target_tip_position": [0.0, 0.02, 0.0]},
        _robot_state(),
    )

    changed_active = [abs(action["section_angles"][index]) > 0.0 for index in (0, 2, 4)]
    assert any(changed_active)
    assert action["section_angles"][1] == pytest.approx(0.0)
    assert action["section_angles"][3] == pytest.approx(0.0)
    assert action["section_angles"][5] == pytest.approx(0.0)
    assert info["status"] == "ok"


def test_pcc_ik_inactive_columns_unchanged() -> None:
    current = [0.0, 0.11, 0.0, -0.22, 0.0, 0.33]
    controller = PccIkController(jacobian=_test_jacobian())

    action, info = controller.compute_action_with_info(
        {"target_tip_position": [0.0, 0.02, 0.0]},
        _robot_state(current),
    )

    assert action["section_angles"][1] == pytest.approx(current[1])
    assert action["section_angles"][3] == pytest.approx(current[3])
    assert action["section_angles"][5] == pytest.approx(current[5])
    assert info["dq"][1] == pytest.approx(0.0)
    assert info["dq"][3] == pytest.approx(0.0)
    assert info["dq"][5] == pytest.approx(0.0)


def test_pcc_ik_grip_and_grasper_rotation_override() -> None:
    controller = PccIkController(jacobian=_test_jacobian())

    action = controller.compute_action(
        {
            "target_tip_position": [0.0, 0.0, 0.0],
            "grip_command": 1.0,
            "grasper_rotation": 0.75,
        },
        _robot_state(),
    )

    assert action["grip_command"] == pytest.approx(1.0)
    assert action["grasper_rotation"] == pytest.approx(0.75)


def test_pcc_ik_clips_section_angles() -> None:
    controller = PccIkController(
        PccIkConfig(max_abs_section_angle=0.2, max_step_norm=10.0),
        jacobian=_test_jacobian(),
    )

    action = controller.compute_action(
        {"target_tip_position": [0.0, 10.0, 0.0]},
        _robot_state([0.19, 0.0, 0.19, 0.0, 0.19, 0.0]),
    )

    assert all(abs(value) <= 0.2 for value in action["section_angles"])


def test_pcc_ik_no_gripper_rotation_key() -> None:
    controller = PccIkController(jacobian=_test_jacobian())

    action = controller.compute_action({"target_tip_position": [0.0, 0.02, 0.0]}, _robot_state())

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert "gripper_rotation" not in action
