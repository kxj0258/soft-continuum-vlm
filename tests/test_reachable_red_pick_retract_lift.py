from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_reachable_red_pick_retract_lift.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "run_reachable_red_pick_retract_lift",
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
        scale=0.68,
    )

    assert result == pytest.approx([1.428, 1.5708, 1.02, 1.5708, 0.68, 1.5708])


def test_build_trial_schedules_has_expected_post_close_trials() -> None:
    module = _load_module()

    schedules = module.build_trial_schedules()

    assert [schedule["name"] for schedule in schedules] == [
        "hold_only",
        "gentle_retract_scale_down",
        "gentle_advance_scale_up",
        "rotation_settle",
        "micro_lift_like",
    ]
    assert [phase["name"] for phase in schedules[0]["phases"][-1:]] == ["post_hold"]
    assert [phase["name"] for phase in schedules[4]["phases"][-2:]] == [
        "lift_like_ramp",
        "lift_like_hold",
    ]


def test_compute_post_close_contact_ratio() -> None:
    module = _load_module()

    assert module.compute_contact_ratio([True, False, True, True]) == pytest.approx(0.75)
    assert module.compute_contact_ratio([]) == pytest.approx(0.0)


def test_compute_red_z_delta() -> None:
    module = _load_module()

    assert module.compute_red_z_delta([0.2, 0.1, 0.331], [0.19, 0.1, 0.337]) == pytest.approx(0.006)


def test_compute_object_follow_score() -> None:
    module = _load_module()

    score = module.compute_object_follow_score(
        post_close_red_grasper_contact_ratio=0.75,
        post_close_red_motion=0.015,
        post_close_red_z_delta=0.004,
        post_close_red_pedestal_contact_ratio=0.25,
        post_close_max_grasper_force=650.0,
    )

    assert score == pytest.approx(3.75)


def test_select_best_trials() -> None:
    module = _load_module()
    trials = [
        {
            "name": "contact",
            "object_follow_score": 3.0,
            "post_close_red_z_delta": 0.001,
            "post_close_red_grasper_contact_ratio": 1.0,
        },
        {
            "name": "z",
            "object_follow_score": 2.0,
            "post_close_red_z_delta": 0.01,
            "post_close_red_grasper_contact_ratio": 0.5,
        },
        {
            "name": "score",
            "object_follow_score": 4.0,
            "post_close_red_z_delta": 0.002,
            "post_close_red_grasper_contact_ratio": 0.75,
        },
    ]

    assert module.select_best_trial(trials, "object_follow_score")["name"] == "score"
    assert module.select_best_trial(trials, "post_close_red_z_delta")["name"] == "z"
    assert module.select_best_trial(trials, "post_close_red_grasper_contact_ratio")["name"] == "contact"
    assert module.select_best_trial([], "object_follow_score") is None


def test_make_judgment_covers_four_paths() -> None:
    module = _load_module()

    assert module.make_judgment(
        trials=[
            {
                "post_close_red_grasper_contact_ratio": 0.5,
                "post_close_red_motion": 0.01,
                "post_close_max_grasper_force": 999.0,
            }
        ]
    ).startswith("[OK]")
    assert module.make_judgment(
        trials=[
            {
                "post_close_red_grasper_contact_ratio": 0.75,
                "post_close_red_motion": 0.005,
                "post_close_max_grasper_force": 200.0,
            }
        ]
    ).startswith("[WARN] contact persisted")
    assert module.make_judgment(
        trials=[
            {
                "post_close_red_grasper_contact_ratio": 0.1,
                "post_close_red_motion": 0.02,
                "post_close_max_grasper_force": 1000.0,
            }
        ]
    ).startswith("[WARN] object motion observed")
    assert module.make_judgment(
        trials=[
            {
                "post_close_red_grasper_contact_ratio": 0.1,
                "post_close_red_motion": 0.001,
                "post_close_max_grasper_force": 200.0,
            }
        ]
    ).startswith("[FAIL]")


def test_validate_action_rejects_gripper_rotation_key() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="gripper_rotation"):
        module.validate_action(
            {
                "section_angles": [1.554, 1.5708, 1.11, 1.5708, 0.74, 1.5708],
                "grip_command": 1.0,
                "gripper_rotation": -0.7854,
            }
        )


def test_summarize_phase_aggregates_contacts_force_and_motion() -> None:
    module = _load_module()
    records = [
        {
            "phase": "lift_like_hold",
            "tip_red_distance": 0.07,
            "red_z": 0.331,
            "has_red_grasper_contact": True,
            "has_red_pedestal_contact": True,
            "has_red_table_contact": False,
            "red_grasper_normal_force": 100.0,
            "red_pedestal_normal_force": 8.0,
            "red_displacement_from_trial_start": 0.02,
        },
        {
            "phase": "lift_like_hold",
            "tip_red_distance": 0.05,
            "red_z": 0.335,
            "has_red_grasper_contact": False,
            "has_red_pedestal_contact": False,
            "has_red_table_contact": True,
            "red_grasper_normal_force": 0.0,
            "red_pedestal_normal_force": 0.0,
            "red_displacement_from_trial_start": 0.03,
        },
        {
            "phase": "other",
            "tip_red_distance": 0.03,
            "red_z": 0.4,
            "has_red_grasper_contact": True,
            "has_red_pedestal_contact": False,
            "has_red_table_contact": False,
            "red_grasper_normal_force": 999.0,
            "red_pedestal_normal_force": 0.0,
            "red_displacement_from_trial_start": 0.2,
        },
    ]

    summary = module.summarize_phase("lift_like_hold", records)

    assert summary["steps"] == 2
    assert summary["min_tip_red_distance"] == pytest.approx(0.05)
    assert summary["final_tip_red_distance"] == pytest.approx(0.05)
    assert summary["red_z_delta"] == pytest.approx(0.004)
    assert summary["has_red_grasper_contact"] is True
    assert summary["red_grasper_contact_ratio"] == pytest.approx(0.5)
    assert summary["red_pedestal_contact_ratio"] == pytest.approx(0.5)
    assert summary["red_table_contact_ratio"] == pytest.approx(0.5)
    assert summary["max_red_grasper_normal_force"] == pytest.approx(100.0)
    assert summary["mean_red_grasper_normal_force"] == pytest.approx(100.0)
    assert summary["max_red_pedestal_normal_force"] == pytest.approx(8.0)
    assert summary["red_displacement_delta"] == pytest.approx(0.01)
