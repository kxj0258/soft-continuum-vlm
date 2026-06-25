from __future__ import annotations

import math

import pytest


def test_tk_control_panel_module_imports() -> None:
    import soft_continuum_vlm.ui.tk_control_panel as tk_control_panel

    assert hasattr(tk_control_panel, "TkFeagineControlPanel")


def _make_panel():
    try:
        from soft_continuum_vlm.ui.tk_control_panel import TkFeagineControlPanel
    except ImportError as exc:
        pytest.skip(f"tkinter unavailable: {exc}")

    try:
        return TkFeagineControlPanel()
    except Exception as exc:
        pytest.skip(f"Tkinter display unavailable: {exc}")


def test_panel_action_schema_radians() -> None:
    panel = _make_panel()
    try:
        action = panel.action()
        assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
        assert "gripper_rotation" not in action
        assert len(action["section_angles"]) == 6
    finally:
        panel.close()


def test_bend_slider_degrees_to_radians() -> None:
    panel = _make_panel()
    try:
        panel.set_section_angle(0, 90.0)
        action = panel.action()
        assert action["section_angles"][0] == pytest.approx(math.pi / 2.0)
    finally:
        panel.close()


def test_direction_slider_degrees_to_radians() -> None:
    panel = _make_panel()
    try:
        panel.set_section_angle(1, -180.0)
        action = panel.action()
        assert action["section_angles"][1] == pytest.approx(-math.pi)
    finally:
        panel.close()


def test_grasper_rotation_degrees_to_radians() -> None:
    panel = _make_panel()
    try:
        panel.set_grasper_rotation(180.0)
        action = panel.action()
        assert action["grasper_rotation"] == pytest.approx(math.pi)
    finally:
        panel.close()


def test_bend_magnitude_clamped_nonnegative() -> None:
    panel = _make_panel()
    try:
        panel.set_section_angle(0, -20.0)
        action = panel.action()
        assert action["section_angles"][0] == pytest.approx(0.0)
    finally:
        panel.close()


def test_presets_output_radians() -> None:
    panel = _make_panel()
    try:
        panel.apply_preset("Bend +Y")
        action = panel.action()
        assert action["section_angles"][0] > 0.0
        assert action["section_angles"][2] > 0.0
        assert action["section_angles"][4] > 0.0
        assert action["section_angles"][1] == pytest.approx(math.pi / 2.0)
        assert action["section_angles"][3] == pytest.approx(math.pi / 2.0)
        assert action["section_angles"][5] == pytest.approx(math.pi / 2.0)
        assert "gripper_rotation" not in action

        panel.apply_preset("Bend -Y")
        action = panel.action()
        assert action["section_angles"][1] == pytest.approx(-math.pi / 2.0)
        assert action["section_angles"][3] == pytest.approx(-math.pi / 2.0)
        assert action["section_angles"][5] == pytest.approx(-math.pi / 2.0)
    finally:
        panel.close()


def test_status_contains_degrees_and_radians() -> None:
    panel = _make_panel()
    try:
        panel.set_section_angle(0, 45.0)
        panel.set_grasper_rotation(90.0)
        status = panel._status_var.get()
        assert "section_angles_deg=" in status
        assert "section_angles_rad=" in status
        assert "grasper_rotation_deg=" in status
        assert "grasper_rotation_rad=" in status
    finally:
        panel.close()


def test_controls_sync_to_action_and_close_is_safe() -> None:
    panel = _make_panel()
    try:
        panel.set_section_angle(0, 90.0)
        panel.set_section_angle_from_text(1, "-180")
        panel.set_grip_command(1.0)
        panel.set_grasper_rotation_from_text("180")
        panel.set_tip_position([0.1, 0.2, 0.3])

        action = panel.action()
        assert action["section_angles"][0] == pytest.approx(math.pi / 2.0)
        assert action["section_angles"][1] == pytest.approx(-math.pi)
        assert action["grip_command"] == pytest.approx(1.0)
        assert action["grasper_rotation"] == pytest.approx(math.pi)
    finally:
        panel.close()
        panel.close()
