from soft_continuum_vlm.controllers.safety_projector import SafetyLimits, SafetyProjector


def test_projector_clips_continuum_and_gripper_fields() -> None:
    projector = SafetyProjector(
        SafetyLimits(
            max_abs_section_angle=0.2,
            max_gripper_rotation=0.5,
            max_contact_force=1.0,
            max_penetration=0.01,
        )
    )
    action = {
        "section_angles": [0.4, -0.3, 0.1, -0.1, 0.0, 0.2],
        "grip_command": 1.5,
        "grasper_rotation": -1.0,
    }
    safe_action, info = projector.project(action)
    assert safe_action == {
        "section_angles": [0.2, -0.2, 0.1, -0.1, 0.0, 0.2],
        "grip_command": 1.0,
        "grasper_rotation": -0.5,
    }
    assert info["clipped"] is True
    assert set(info["clipped_fields"]) == {"section_angles", "grip_command", "grasper_rotation"}


def test_projector_freezes_motion_when_contact_limits_are_exceeded() -> None:
    projector = SafetyProjector(
        SafetyLimits(
            max_abs_section_angle=0.2,
            max_gripper_rotation=0.5,
            max_contact_force=1.0,
            max_penetration=0.01,
        )
    )
    action = {
        "section_angles": [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
        "grip_command": 0.4,
        "grasper_rotation": 0.2,
    }
    safe_action, info = projector.project(action, contact_force=2.0, penetration=0.02)
    assert safe_action == {"grip_command": 0.4}
    assert info["contact_limited"] is True
    assert info["penetration_limited"] is True
    assert set(info["blocked_fields"]) == {"section_angles", "grasper_rotation"}
