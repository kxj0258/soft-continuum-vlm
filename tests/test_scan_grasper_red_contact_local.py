from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "scan_grasper_red_contact_local.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "scan_grasper_red_contact_local",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_action_rejects_gripper_rotation_key() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="gripper_rotation"):
        module.validate_action(
            {
                "section_angles": [0.35, 1.5708, 0.25, 1.5708, 0.15, 1.5708],
                "grip_command": 0.0,
                "gripper_rotation": 0.0,
            }
        )


def test_build_local_commands_respects_max_commands_and_section_angle_shape() -> None:
    module = _load_module()
    commands = module.build_local_commands(max_commands=73)

    assert len(commands) <= 73
    assert len(commands) > 0
    for command in commands:
        assert set(command) == {"name", "section_angles", "grip_command", "grasper_rotation"}
        assert len(command["section_angles"]) == 6


def test_build_local_commands_keep_magnitudes_and_directions_in_range() -> None:
    module = _load_module()
    commands = module.build_local_commands(max_commands=120)

    for command in commands:
        section_angles = command["section_angles"]
        for index in (0, 2, 4):
            assert 0.0 <= float(section_angles[index]) <= 2.356
        for index in (1, 3, 5):
            assert -math.pi <= float(section_angles[index]) <= math.pi


def test_contact_flags_detect_red_grasper_contact() -> None:
    module = _load_module()
    flags = module.contact_flags(
        [
            {
                "geom1": "red_object_geom",
                "geom2": "finger_1",
                "body1": "red_object",
                "body2": "feagine_grasper_tip",
            }
        ]
    )

    assert flags["has_red_grasper_contact"] is True
    assert flags["has_tip_contact"] is True
    assert flags["has_red_object_contact"] is True


def test_choose_best_command_prefers_smallest_distance_then_contact() -> None:
    module = _load_module()
    summaries = [
        {
            "name": "a",
            "action": {"section_angles": [0.35, 1.5708, 0.25, 1.5708, 0.15, 1.5708], "grip_command": 0.0, "grasper_rotation": 0.0},
            "initial_tip_red_distance": 0.5,
            "min_tip_red_distance": 0.22,
            "final_tip_red_distance": 0.28,
            "distance_reduction_ratio": 0.56,
            "has_red_grasper_contact": False,
        },
        {
            "name": "b",
            "action": {"section_angles": [0.5, 1.5708, 0.35, 1.5708, 0.2, 1.5708], "grip_command": 1.0, "grasper_rotation": 0.0},
            "initial_tip_red_distance": 0.5,
            "min_tip_red_distance": 0.18,
            "final_tip_red_distance": 0.24,
            "distance_reduction_ratio": 0.64,
            "has_red_grasper_contact": True,
        },
    ]

    best = module.choose_best_command(summaries)

    assert best["name"] == "b"
    assert best["min_tip_red_distance"] == pytest.approx(0.18)


def test_make_judgment_covers_ok_warn_and_fail_paths() -> None:
    module = _load_module()

    assert module.make_judgment(
        any_red_grasper_contact=True,
        best_min_tip_red_distance=0.2,
        best_distance_reduction_ratio=0.1,
    ).startswith("[OK]")
    assert module.make_judgment(
        any_red_grasper_contact=False,
        best_min_tip_red_distance=0.09,
        best_distance_reduction_ratio=0.1,
    ).startswith("[WARN] grasper reached near")
    assert module.make_judgment(
        any_red_grasper_contact=False,
        best_min_tip_red_distance=0.2,
        best_distance_reduction_ratio=0.35,
    ).startswith("[WARN] grasper approached")
    assert module.make_judgment(
        any_red_grasper_contact=False,
        best_min_tip_red_distance=0.2,
        best_distance_reduction_ratio=0.1,
    ).startswith("[FAIL]")
