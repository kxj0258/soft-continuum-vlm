from __future__ import annotations

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController, zero_feagine_action
from soft_continuum_vlm.controllers.safety_projector import SafetyLimits, SafetyProjector
from soft_continuum_vlm.controllers.scripted_expert import ScriptedExpert


def test_zero_feagine_action_uses_runtime_control_names() -> None:
    action = zero_feagine_action(section_count=3)

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert action["section_angles"] == [0.0] * 6


def test_scripted_expert_returns_safety_projected_feagine_action() -> None:
    expert = ScriptedExpert(
        controller=PccIkController(section_count=3),
        safety_projector=SafetyProjector(
            SafetyLimits(
                max_abs_section_angle=0.2,
                max_gripper_rotation=0.5,
                max_contact_force=1.0,
                max_penetration=0.01,
            )
        ),
    )

    action, info = expert.act(
        {
            "robot_state": {"section_angles": [0.0] * 6},
            "contact": {"max_force": 0.0, "max_penetration": 0.0, "contacts": []},
        }
    )

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert info["source"] == "scripted_pcc_ik"
    assert info["safety"]["clipped"] is False
