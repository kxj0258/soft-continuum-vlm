from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pytest

from soft_continuum_vlm.controllers.scripted_expert import ScriptedExpert


class DummyPccController:
    def __init__(self, section_angles: list[float] | None = None) -> None:
        self.calls: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        self.section_angles = section_angles

    def compute_action_with_info(
        self,
        target_state: Mapping[str, Any],
        robot_state: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append((dict(target_state), dict(robot_state)))
        section_angles = self.section_angles or list(robot_state.get("section_angles", [0.0] * 6))
        return {
            "section_angles": section_angles,
            "grip_command": float(target_state.get("grip_command", robot_state.get("grip_command", 0.0))),
            "grasper_rotation": float(
                target_state.get("grasper_rotation", robot_state.get("grasper_rotation", 0.0))
            ),
        }, {
            "source": "dummy_pcc",
            "status": "ok",
            "dq": [0.0] * len(section_angles),
        }


def _observation(tip: list[float] | None = None, section_angles: list[float] | None = None) -> dict[str, Any]:
    return {
        "robot_state": {
            "tip_pose": {"position": tip if tip is not None else [0.0, 0.0, 0.643]},
            "section_angles": section_angles if section_angles is not None else [0.0] * 6,
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        "contact": {"max_force": 0.0, "max_penetration": 0.0, "contacts": []},
    }


def test_pick_expert_creates_virtual_target() -> None:
    expert = ScriptedExpert(pcc_controller=DummyPccController(), target_offset=(0.0, 0.02, 0.0))

    _action, info = expert.act(_observation([0.0, 0.0, 0.643]))

    assert info["red_object_position"] == pytest.approx([0.0, 0.02, 0.643])
    assert info["target_source"] == "initial_tip_plus_offset"


def test_pick_expert_action_schema() -> None:
    expert = ScriptedExpert(pcc_controller=DummyPccController())

    action, info = expert.act(_observation())

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert "gripper_rotation" not in action
    assert "gripper_rotation" not in json.dumps(info)


def test_pick_expert_phase_progression_close() -> None:
    expert = ScriptedExpert(
        pcc_controller=DummyPccController(),
        target_offset=(0.0, 0.02, 0.0),
        position_tolerance=0.006,
    )
    expert.act(_observation([0.0, 0.0, 0.643]))

    _action, info = expert.act(_observation([0.0, 0.019, 0.643]))

    assert info["phase"] == "close_gripper"
    assert info["phase_transition"]["changed"] is True
    assert info["phase_transition"]["to"] == "close_gripper"


def test_pick_expert_anchor_is_recorded_on_close() -> None:
    anchor_angles = [0.1, 0.0, 0.05, 0.0, 0.02, 0.0]
    expert = ScriptedExpert(
        pcc_controller=DummyPccController(),
        target_offset=(0.0, 0.02, 0.0),
        position_tolerance=0.006,
    )
    expert.act(_observation([0.0, 0.0, 0.643]))

    _action, info = expert.act(_observation([0.0, 0.019, 0.643], section_angles=anchor_angles))

    assert info["phase"] == "close_gripper"
    assert info["grasp_anchor_position"] == pytest.approx([0.0, 0.019, 0.643])
    assert info["grasp_anchor_section_angles"] == pytest.approx(anchor_angles)


def test_pick_expert_close_loop_hold_uses_pcc() -> None:
    anchor_angles = [0.1, 0.0, 0.05, 0.0, 0.02, 0.0]
    pcc = DummyPccController(section_angles=anchor_angles)
    expert = ScriptedExpert(
        pcc_controller=pcc,
        target_offset=(0.0, 0.02, 0.0),
        position_tolerance=0.006,
    )
    expert.act(_observation([0.0, 0.0, 0.643]))

    action, info = expert.act(_observation([0.0, 0.019, 0.643], section_angles=anchor_angles))

    assert info["phase"] == "close_gripper"
    assert len(pcc.calls) == 2
    assert pcc.calls[-1][0]["target_tip_position"] == pytest.approx([0.0, 0.02, 0.643])
    assert pcc.calls[-1][0]["grip_command"] == pytest.approx(1.0)
    assert action["grip_command"] == pytest.approx(1.0)


def test_pick_expert_close_loop_hold_blends_action() -> None:
    current_angles = [0.1, 0.0, 0.05, 0.0, 0.02, 0.0]
    raw_angles = [0.2, 0.0, 0.15, 0.0, 0.12, 0.0]
    expert = ScriptedExpert(
        pcc_controller=DummyPccController(section_angles=raw_angles),
        target_offset=(0.0, 0.02, 0.0),
        position_tolerance=0.006,
        close_hold_position_gain_scale=0.5,
    )
    expert.act(_observation([0.0, 0.0, 0.643]))

    action, info = expert.act(_observation([0.0, 0.019, 0.643], section_angles=current_angles))

    assert info["phase"] == "close_gripper"
    assert action["section_angles"] == pytest.approx([0.15, 0.0, 0.1, 0.0, 0.07, 0.0])


def test_pick_expert_overshoot_guard_holds_current() -> None:
    current_angles = [0.3, 0.0, 0.2, 0.0, 0.1, 0.0]
    expert = ScriptedExpert(
        pcc_controller=DummyPccController(section_angles=[0.8] * 6),
        target_offset=(0.0, 0.02, 0.0),
        position_tolerance=0.006,
        max_post_close_drift=0.02,
        close_steps=5,
    )
    expert.act(_observation([0.0, 0.0, 0.643]))
    expert.act(_observation([0.0, 0.019, 0.643], section_angles=[0.1] * 6))

    action, info = expert.act(_observation([0.0, 0.055, 0.643], section_angles=current_angles))

    assert info["phase"] == "close_gripper"
    assert info["overshoot_guard_triggered"] is True
    assert action["section_angles"] == pytest.approx(current_angles)
    assert action["grip_command"] == pytest.approx(1.0)


def test_pick_expert_hold_closed_does_not_lift() -> None:
    anchor_angles = [0.1, 0.0, 0.05, 0.0, 0.02, 0.0]
    pcc = DummyPccController()
    expert = ScriptedExpert(
        pcc_controller=pcc,
        target_offset=(0.0, 0.02, 0.0),
        position_tolerance=0.006,
        close_steps=1,
    )
    expert.act(_observation([0.0, 0.0, 0.643]))
    expert.act(_observation([0.0, 0.019, 0.643], section_angles=anchor_angles))

    action, info = expert.act(_observation([0.0, 0.019, 0.643], section_angles=[0.0] * 6))

    assert info["phase"] == "hold_closed"
    assert info["target_state"]["target_tip_position"] == pytest.approx([0.0, 0.02, 0.643])
    assert info["target_state"]["target_tip_position"][2] == pytest.approx(0.643)
    assert action["grip_command"] == pytest.approx(1.0)
    assert len(pcc.calls) == 3


def test_pick_expert_uses_pcc_controller() -> None:
    pcc = DummyPccController()
    expert = ScriptedExpert(pcc_controller=pcc)

    expert.act(_observation())

    assert len(pcc.calls) == 1


def test_pick_expert_info_json_serializable() -> None:
    expert = ScriptedExpert(pcc_controller=DummyPccController())

    _action, info = expert.act(_observation())

    json.dumps(info)


def test_done_closed_loop_mode_uses_pcc() -> None:
    pcc = DummyPccController(section_angles=[0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
    expert = ScriptedExpert(
        pcc_controller=pcc,
        done_hold_mode="closed_loop",
        done_hold_position_gain_scale=0.2,
        done_max_section_step_norm=0.1,
        done_overshoot_guard=True,
        done_overshoot_margin=0.003,
    )
    expert.phase = "done"

    action, info = expert.act(_observation([0.0, 0.0, 0.643]))

    assert len(pcc.calls) == 1
    assert action["grip_command"] == pytest.approx(1.0)
    assert info["done_hold_mode"] == "closed_loop"
    assert info["done_continue_control"] is True


def test_done_closed_loop_blends_action() -> None:
    pcc = DummyPccController(section_angles=[0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
    expert = ScriptedExpert(
        pcc_controller=pcc,
        done_hold_mode="closed_loop",
        done_hold_position_gain_scale=0.2,
        done_max_section_step_norm=0.1,
    )
    expert.phase = "done"

    action, info = expert.act(_observation([0.0, 0.0, 0.643], section_angles=[0.0] * 6))

    assert action["section_angles"] == pytest.approx([0.02, 0.0, 0.02, 0.0, 0.02, 0.0])
    assert info["done_hold_position_gain_scale"] == pytest.approx(0.2)


def test_done_closed_loop_step_norm_limit() -> None:
    pcc = DummyPccController(section_angles=[1.0] * 6)
    expert = ScriptedExpert(
        pcc_controller=pcc,
        done_hold_mode="closed_loop",
        done_hold_position_gain_scale=1.0,
        done_max_section_step_norm=0.03,
    )
    expert.phase = "done"

    action, _info = expert.act(_observation([0.0, 0.0, 0.643], section_angles=[0.0] * 6))

    delta = np.asarray(action["section_angles"], dtype=np.float64)
    assert float(np.linalg.norm(delta)) <= 0.03 + 1e-9


def test_done_overshoot_guard_prevents_positive_push() -> None:
    current_angles = [0.3, 0.0, 0.2, 0.0, 0.1, 0.0]
    pcc = DummyPccController(section_angles=[0.8] * 6)
    expert = ScriptedExpert(
        pcc_controller=pcc,
        done_hold_mode="closed_loop",
        done_hold_position_gain_scale=1.0,
        done_max_section_step_norm=1.0,
        done_overshoot_guard=True,
        done_overshoot_margin=0.003,
    )
    expert.phase = "done"

    observation = _observation([0.0, 0.055, 0.643], section_angles=current_angles)
    observation["virtual_objects"] = {
        "red_object": {"pose": {"position": [0.0, 0.02, 0.643]}, "available": True}
    }

    action, info = expert.act(observation)

    assert info["done_overshoot_guard_triggered"] is True
    assert action["section_angles"] == pytest.approx(current_angles)
    assert action["grip_command"] == pytest.approx(1.0)


def test_done_action_schema_no_gripper_rotation() -> None:
    expert = ScriptedExpert(pcc_controller=DummyPccController(), done_hold_mode="closed_loop")
    expert.phase = "done"

    action, info = expert.act(_observation())

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert "gripper_rotation" not in action
    assert "gripper_rotation" not in json.dumps(info)
