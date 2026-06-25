from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_reachable_red_pick_attempt.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "run_reachable_red_pick_attempt",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_preclose_section_angles_scales_only_bend_magnitudes() -> None:
    module = _load_module()

    result = module.build_preclose_section_angles(
        full_section_angles=[2.1, 1.5708, 1.5, 1.5708, 1.0, 1.5708],
        preclose_scale=0.7,
    )

    assert result == pytest.approx([1.47, 1.5708, 1.05, 1.5708, 0.7, 1.5708])


def test_validate_action_rejects_gripper_rotation_key() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="gripper_rotation"):
        module.validate_action(
            {
                "section_angles": [1.47, 1.5708, 1.05, 1.5708, 0.7, 1.5708],
                "grip_command": 0.0,
                "gripper_rotation": -0.7854,
            }
        )


def test_build_phase_schedule_contains_required_phases() -> None:
    module = _load_module()

    schedule = module.build_phase_schedule(include_relax=False)

    assert [phase["name"] for phase in schedule] == [
        "approach_ramp",
        "hold_preclose",
        "close_ramp",
        "hold_closed",
    ]


def test_contact_ratio_computation() -> None:
    module = _load_module()

    ratio = module.compute_contact_ratio([True, False, True, True])

    assert ratio == pytest.approx(0.75)


def test_success_flags_computation() -> None:
    module = _load_module()

    flags = module.compute_success_flags(
        hold_preclose_has_contact=False,
        close_has_contact=True,
        hold_closed_has_contact=True,
        hold_closed_contact_ratio=0.4,
        red_displacement_after_close=0.01,
    )

    assert flags == {
        "clean_preclose": True,
        "close_added_contact": True,
        "hold_closed_contact_persistent": True,
        "red_object_moved_after_close": True,
    }


def test_make_judgment_covers_four_paths() -> None:
    module = _load_module()

    assert module.make_judgment(
        {
            "clean_preclose": True,
            "close_added_contact": True,
            "hold_closed_contact_persistent": True,
            "red_object_moved_after_close": True,
        }
    ).startswith("[OK]")
    assert module.make_judgment(
        {
            "clean_preclose": True,
            "close_added_contact": True,
            "hold_closed_contact_persistent": False,
            "red_object_moved_after_close": True,
        }
    ).startswith("[WARN] close contact occurred")
    assert module.make_judgment(
        {
            "clean_preclose": False,
            "close_added_contact": True,
            "hold_closed_contact_persistent": True,
            "red_object_moved_after_close": True,
        }
    ).startswith("[WARN] preclose already had contact")
    assert module.make_judgment(
        {
            "clean_preclose": True,
            "close_added_contact": False,
            "hold_closed_contact_persistent": False,
            "red_object_moved_after_close": False,
        }
    ).startswith("[FAIL]")


def test_phase_summary_aggregation() -> None:
    module = _load_module()
    records = [
        {
            "phase": "close_ramp",
            "tip_red_distance": 0.08,
            "red_displacement": 0.002,
            "has_red_grasper_contact": False,
            "red_grasper_normal_force": 0.0,
            "contact_pairs": [],
        },
        {
            "phase": "close_ramp",
            "tip_red_distance": 0.05,
            "red_displacement": 0.010,
            "has_red_grasper_contact": True,
            "red_grasper_normal_force": 5.0,
            "contact_pairs": ["<unnamed:42> <-> red_object_geom"],
        },
        {
            "phase": "close_ramp",
            "tip_red_distance": 0.06,
            "red_displacement": 0.012,
            "has_red_grasper_contact": True,
            "red_grasper_normal_force": 3.0,
            "contact_pairs": ["<unnamed:48> <-> red_object_geom"],
        },
    ]

    summary = module.summarize_phase("close_ramp", records)

    assert summary["steps"] == 3
    assert summary["min_tip_red_distance"] == pytest.approx(0.05)
    assert summary["final_tip_red_distance"] == pytest.approx(0.06)
    assert summary["contact_step_count"] == 2
    assert summary["contact_ratio"] == pytest.approx(2.0 / 3.0)
    assert summary["max_red_grasper_normal_force"] == pytest.approx(5.0)
    assert summary["mean_red_grasper_normal_force"] == pytest.approx(4.0)
