from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "tune_reachable_red_pick_parameters.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "tune_reachable_red_pick_parameters",
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
        scale=0.66,
    )

    assert result == pytest.approx([1.386, 1.5708, 0.99, 1.5708, 0.66, 1.5708])


def test_build_trial_grid_has_expected_count_and_values() -> None:
    module = _load_module()

    grid = module.build_trial_grid(max_trials=135)

    assert len(grid) == 135
    assert grid[0]["preclose_scale"] == pytest.approx(0.58)
    assert grid[0]["approach_rotation"] == pytest.approx(0.0)
    assert grid[0]["close_rotation"] == pytest.approx(-0.3927)
    assert grid[0]["close_advance_scale"] == pytest.approx(0.0)
    assert grid[-1]["preclose_scale"] == pytest.approx(0.74)
    assert grid[-1]["approach_rotation"] == pytest.approx(-0.7854)
    assert grid[-1]["close_rotation"] == pytest.approx(-1.1781)
    assert grid[-1]["close_advance_scale"] == pytest.approx(0.06)


def test_validate_action_rejects_gripper_rotation_key() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="gripper_rotation"):
        module.validate_action(
            {
                "section_angles": [1.386, 1.5708, 0.99, 1.5708, 0.66, 1.5708],
                "grip_command": 0.0,
                "gripper_rotation": -0.7854,
            }
        )


def test_compute_trial_flags() -> None:
    module = _load_module()

    flags = module.compute_trial_flags(
        hold_preclose_contact_ratio=0.02,
        close_ramp_contact_ratio=0.1,
        hold_closed_contact_ratio=0.3,
        red_motion_after_close=0.01,
        max_red_grasper_normal_force=450.0,
    )

    assert flags == {
        "clean_preclose": True,
        "close_added_contact": True,
        "persistent_post_close": True,
        "red_object_moved_after_close": True,
        "force_not_extreme": True,
    }


def test_compute_trial_score_penalizes_extreme_force_and_dirty_preclose() -> None:
    module = _load_module()

    score = module.compute_trial_score(
        clean_preclose=False,
        close_added_contact=True,
        persistent_post_close=True,
        red_object_moved_after_close=True,
        force_not_extreme=False,
        min_tip_red_distance=0.05,
        max_red_grasper_normal_force=1100.0,
        hold_preclose_contact_ratio=0.10,
    )

    assert score == pytest.approx(-1.25)


def test_rank_trials_prefers_higher_score() -> None:
    module = _load_module()

    ranked = module.rank_trials(
        [
            {"name": "b", "score": 2.0, "clean_preclose": False, "persistent_post_close": False},
            {"name": "a", "score": 5.0, "clean_preclose": True, "persistent_post_close": True},
        ]
    )

    assert [trial["name"] for trial in ranked] == ["a", "b"]


def test_make_judgment_covers_four_paths() -> None:
    module = _load_module()

    assert module.make_judgment(
        best_clean_persistent_trial={"name": "ok"},
        any_clean_preclose=True,
        any_close_added_contact=True,
        any_persistent_post_close=True,
        any_clean_persistent_contact=True,
    ).startswith("[OK]")
    assert module.make_judgment(
        best_clean_persistent_trial=None,
        any_clean_preclose=True,
        any_close_added_contact=True,
        any_persistent_post_close=False,
        any_clean_persistent_contact=False,
    ).startswith("[WARN] found clean close contact")
    assert module.make_judgment(
        best_clean_persistent_trial=None,
        any_clean_preclose=False,
        any_close_added_contact=True,
        any_persistent_post_close=True,
        any_clean_persistent_contact=False,
    ).startswith("[WARN] contact persists")
    assert module.make_judgment(
        best_clean_persistent_trial=None,
        any_clean_preclose=False,
        any_close_added_contact=False,
        any_persistent_post_close=False,
        any_clean_persistent_contact=False,
    ).startswith("[FAIL]")


def test_select_best_clean_persistent_trial() -> None:
    module = _load_module()

    best = module.select_best_clean_persistent_trial(
        [
            {
                "name": "dirty",
                "score": 9.0,
                "clean_preclose": False,
                "close_added_contact": True,
                "persistent_post_close": True,
                "force_not_extreme": True,
            },
            {
                "name": "clean",
                "score": 7.0,
                "clean_preclose": True,
                "close_added_contact": True,
                "persistent_post_close": True,
                "force_not_extreme": True,
            },
        ]
    )

    assert best is not None
    assert best["name"] == "clean"
