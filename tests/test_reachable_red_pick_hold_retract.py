from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_reachable_red_pick_hold_retract.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "run_reachable_red_pick_hold_retract",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scale_section_angles_scales_only_bend_magnitudes() -> None:
    module = _load_module()

    result = module.scale_section_angles(
        [2.1, 1.5708, 1.5, 1.5708, 1.0, 1.5708],
        scale=0.7,
    )

    assert result == pytest.approx([1.47, 1.5708, 1.05, 1.5708, 0.7, 1.5708])


def test_build_trial_schedules_contains_required_trial_names() -> None:
    module = _load_module()

    schedules = module.build_trial_schedules()

    assert [schedule["name"] for schedule in schedules] == [
        "extended_hold",
        "gentle_retract_downscale",
        "gentle_advance_upscale",
        "regrip_squeeze",
    ]


def test_compute_post_close_contact_ratio() -> None:
    module = _load_module()

    ratio = module.compute_post_close_contact_ratio([True, False, True, True])

    assert ratio == pytest.approx(0.75)


def test_compute_red_motion_after_close_returns_difference() -> None:
    module = _load_module()

    motion = module.compute_red_motion_after_close(0.019, 0.004)

    assert motion == pytest.approx(0.015)


def test_choose_best_trial_uses_requested_metric() -> None:
    module = _load_module()

    best = module.choose_best_trial(
        [
            {
                "name": "a",
                "post_close_contact_ratio": 0.10,
                "red_motion_after_close": 0.005,
                "post_close_max_force": 1.0,
                "overall_min_tip_red_distance": 0.06,
            },
            {
                "name": "b",
                "post_close_contact_ratio": 0.25,
                "red_motion_after_close": 0.003,
                "post_close_max_force": 0.5,
                "overall_min_tip_red_distance": 0.05,
            },
        ],
        "post_close_contact_ratio",
    )

    assert best is not None
    assert best["name"] == "b"


def test_make_judgment_covers_four_paths() -> None:
    module = _load_module()

    assert module.make_judgment(
        any_post_close_contact=True,
        max_post_close_contact_ratio=0.3,
        max_red_motion_after_close=0.01,
        any_intermittent_contact=True,
    ).startswith("[OK]")
    assert module.make_judgment(
        any_post_close_contact=True,
        max_post_close_contact_ratio=0.2,
        max_red_motion_after_close=0.01,
        any_intermittent_contact=True,
    ).startswith("[WARN] contact occurred")
    assert module.make_judgment(
        any_post_close_contact=False,
        max_post_close_contact_ratio=0.0,
        max_red_motion_after_close=0.01,
        any_intermittent_contact=True,
    ).startswith("[WARN] object moved")
    assert module.make_judgment(
        any_post_close_contact=False,
        max_post_close_contact_ratio=0.0,
        max_red_motion_after_close=0.001,
        any_intermittent_contact=False,
    ).startswith("[FAIL]")


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


def test_summarize_phase_aggregates_contact_ratio_force_and_motion() -> None:
    module = _load_module()

    summary = module.summarize_phase(
        "hold_closed_initial",
        [
            {
                "phase": "hold_closed_initial",
                "tip_red_distance": 0.08,
                "red_displacement_from_trial_start": 0.002,
                "has_red_grasper_contact": False,
                "red_grasper_normal_force": 0.0,
            },
            {
                "phase": "hold_closed_initial",
                "tip_red_distance": 0.05,
                "red_displacement_from_trial_start": 0.010,
                "has_red_grasper_contact": True,
                "red_grasper_normal_force": 5.0,
            },
            {
                "phase": "hold_closed_initial",
                "tip_red_distance": 0.06,
                "red_displacement_from_trial_start": 0.012,
                "has_red_grasper_contact": True,
                "red_grasper_normal_force": 3.0,
            },
        ],
    )

    assert summary["steps"] == 3
    assert summary["min_tip_red_distance"] == pytest.approx(0.05)
    assert summary["final_tip_red_distance"] == pytest.approx(0.06)
    assert summary["contact_step_count"] == 2
    assert summary["contact_ratio"] == pytest.approx(2.0 / 3.0)
    assert summary["max_red_grasper_normal_force"] == pytest.approx(5.0)
    assert summary["mean_red_grasper_normal_force"] == pytest.approx(4.0)
    assert summary["red_displacement_delta"] == pytest.approx(0.01)
