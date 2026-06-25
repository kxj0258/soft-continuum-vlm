from __future__ import annotations

import pytest

from soft_continuum_vlm.ui.manual_control import ManualControlConfig, ManualFeagineController


def test_manual_action_schema() -> None:
    controller = ManualFeagineController()

    action = controller.action()

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert "gripper_rotation" not in action


def test_manual_section_angle_keys() -> None:
    controller = ManualFeagineController()

    controller.handle_key("1")
    assert controller.action()["section_angles"][0] == pytest.approx(0.03)

    controller.handle_key("q")
    assert controller.action()["section_angles"][0] == pytest.approx(0.0)


def test_manual_inactive_axes_still_controllable() -> None:
    controller = ManualFeagineController()

    controller.handle_key("a")
    controller.handle_key("s")
    controller.handle_key("d")

    action = controller.action()
    assert action["section_angles"][1] == pytest.approx(0.03)
    assert action["section_angles"][3] == pytest.approx(0.03)
    assert action["section_angles"][5] == pytest.approx(0.03)


def test_manual_gripper_open_close() -> None:
    controller = ManualFeagineController()

    controller.handle_key("p")
    assert controller.action()["grip_command"] == pytest.approx(1.0)

    controller.handle_key("o")
    assert controller.action()["grip_command"] == pytest.approx(0.0)


def test_manual_reset() -> None:
    controller = ManualFeagineController()

    controller.handle_key("1")
    controller.handle_key("p")
    controller.handle_key("r")
    controller.handle_key("0")

    action = controller.action()
    assert action["section_angles"] == pytest.approx([0.0] * 6)
    assert action["grip_command"] == pytest.approx(0.0)
    assert action["grasper_rotation"] == pytest.approx(0.0)


def test_manual_clip() -> None:
    controller = ManualFeagineController(
        ManualControlConfig(section_step=0.3, max_abs_section_angle=0.8)
    )

    for _ in range(5):
        controller.handle_key("1")

    assert controller.action()["section_angles"][0] == pytest.approx(0.8)
