from soft_continuum_vlm.controllers.safety_projector import SafetyLimits, SafetyProjector


def test_projector_clips_continuum_and_gripper_fields() -> None:
    projector = SafetyProjector(
        SafetyLimits(
            max_delta_kappa=0.2,
            max_delta_length=0.05,
            max_gripper_rotation=0.5,
            max_contact_force=1.0,
            max_penetration=0.01,
        )
    )
    action = {
        "delta_kappa_x": 0.4,
        "delta_kappa_y": -0.3,
        "delta_length": 0.2,
        "gripper_open": 1.5,
        "gripper_rotation": -1.0,
    }
    safe_action, info = projector.project(action)
    assert safe_action == {
        "delta_kappa_x": 0.2,
        "delta_kappa_y": -0.2,
        "delta_length": 0.05,
        "gripper_open": 1.0,
        "gripper_rotation": -0.5,
    }
    assert info["clipped"] is True
    assert set(info["clipped_fields"]) == {
        "delta_kappa_x",
        "delta_kappa_y",
        "delta_length",
        "gripper_open",
        "gripper_rotation",
    }


def test_projector_freezes_motion_when_contact_limits_are_exceeded() -> None:
    projector = SafetyProjector(
        SafetyLimits(
            max_delta_kappa=0.2,
            max_delta_length=0.05,
            max_gripper_rotation=0.5,
            max_contact_force=1.0,
            max_penetration=0.01,
        )
    )
    action = {
        "delta_kappa_x": 0.1,
        "delta_kappa_y": 0.1,
        "delta_length": 0.02,
        "gripper_open": 0.4,
        "gripper_rotation": 0.2,
    }
    safe_action, info = projector.project(action, contact_force=2.0, penetration=0.02)
    assert safe_action["delta_kappa_x"] == 0.0
    assert safe_action["delta_kappa_y"] == 0.0
    assert safe_action["delta_length"] == 0.0
    assert safe_action["gripper_open"] == 0.4
    assert safe_action["gripper_rotation"] == 0.0
    assert info["contact_limited"] is True
    assert info["penetration_limited"] is True
