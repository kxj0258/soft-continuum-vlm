from __future__ import annotations

import pytest

from soft_continuum_vlm.controllers.safety_projector import SafetyLimits, SafetyProjector


def projector() -> SafetyProjector:
    return SafetyProjector(
        SafetyLimits(
            max_abs_section_angle=1.0,
            max_gripper_rotation=1.0,
            max_contact_force=0.5,
            max_penetration=0.01,
        )
    )


def test_hold_current_mode_freezes_blocked_fields_to_current_robot_state() -> None:
    action = {
        "section_angles": [0.5] * 6,
        "grip_command": 0.7,
        "grasper_rotation": 0.8,
    }

    safe_action, info = projector().project(
        action,
        contact_force=2.0,
        safety_mode="hold_current",
        current_robot_state={"section_angles": [0.1] * 6, "grasper_rotation": -0.2},
    )

    assert safe_action["section_angles"] == [0.1] * 6
    assert safe_action["grip_command"] == pytest.approx(0.7)
    assert safe_action["grasper_rotation"] == pytest.approx(-0.2)
    assert info["safety_mode"] == "hold_current"
    assert set(info["blocked_fields"]) == {"section_angles", "grasper_rotation"}


def test_scale_down_mode_scales_motion_from_current_robot_state() -> None:
    action = {
        "section_angles": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "grip_command": 0.0,
        "grasper_rotation": 1.0,
    }

    safe_action, info = projector().project(
        action,
        penetration=0.02,
        safety_mode="scale_down",
        current_robot_state={"section_angles": [0.0] * 6, "grasper_rotation": 0.0},
    )

    assert safe_action["section_angles"] == pytest.approx([0.2, 0.0, 0.2, 0.0, 0.2, 0.0])
    assert safe_action["grasper_rotation"] == pytest.approx(0.2)
    assert info["safety_mode"] == "scale_down"
