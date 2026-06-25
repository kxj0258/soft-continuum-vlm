from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "scan_grasper_red_contact.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "scan_grasper_red_contact",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_module_imports() -> None:
    module = _load_module()
    assert hasattr(module, "build_commands")
    assert hasattr(module, "choose_best_command")
    assert hasattr(module, "make_judgment")


def test_build_commands_contains_required_actions() -> None:
    module = _load_module()
    commands = module.build_commands(include_strong=True)

    names = [command["name"] for command in commands]
    assert "straight_open" in names
    assert "bend_diag_to_red" in names
    assert "bend_diag_to_red_strong" in names
    assert "bend_diag_to_red_close" in names

    for command in commands:
        assert set(command) == {"name", "section_angles", "grip_command", "grasper_rotation"}
        assert len(command["section_angles"]) == 6
        assert "gripper_rotation" not in command


def test_detect_contact_flags_for_red_grasper_and_tip() -> None:
    module = _load_module()
    contacts = [
        {
            "geom1": "red_object_geom",
            "geom2": "finger_1",
            "body1": "red_object",
            "body2": "feagine_grasper_tip",
        }
    ]

    flags = module.contact_flags(contacts)

    assert flags["has_red_grasper_contact"] is True
    assert flags["has_tip_contact"] is True
    assert flags["has_red_object_contact"] is True


def test_choose_best_command_prefers_smallest_distance() -> None:
    module = _load_module()
    summaries = [
        {
            "name": "a",
            "initial_tip_red_distance": 1.0,
            "min_tip_red_distance": 0.8,
            "final_tip_red_distance": 0.9,
            "has_red_grasper_contact": False,
            "red_object_displacement": 0.0,
            "contact_pairs_seen": [],
        },
        {
            "name": "b",
            "initial_tip_red_distance": 1.0,
            "min_tip_red_distance": 0.4,
            "final_tip_red_distance": 0.5,
            "has_red_grasper_contact": False,
            "red_object_displacement": 0.01,
            "contact_pairs_seen": [],
        },
    ]

    best = module.choose_best_command(summaries)

    assert best["name"] == "b"
    assert best["min_tip_red_distance"] == pytest.approx(0.4)


def test_make_judgment_uses_contact_then_approach_ratio() -> None:
    module = _load_module()

    assert module.make_judgment(
        any_red_grasper_contact=True,
        best_initial_tip_red_distance=1.0,
        best_min_tip_red_distance=0.95,
    ).startswith("[OK]")

    assert module.make_judgment(
        any_red_grasper_contact=False,
        best_initial_tip_red_distance=1.0,
        best_min_tip_red_distance=0.79,
    ).startswith("[WARN]")

    assert module.make_judgment(
        any_red_grasper_contact=False,
        best_initial_tip_red_distance=1.0,
        best_min_tip_red_distance=0.95,
    ).startswith("[FAIL]")
