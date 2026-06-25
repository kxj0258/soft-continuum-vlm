from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sweep_preclose_gripper_close_contact.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "sweep_preclose_gripper_close_contact",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scale_section_angles_only_scales_bend_magnitudes() -> None:
    module = _load_module()

    scaled = module.scale_section_angles(
        [2.1, 1.5708, 1.5, 1.5708, 1.0, 1.5708],
        scale=0.8,
    )

    assert scaled == pytest.approx([1.68, 1.5708, 1.2, 1.5708, 0.8, 1.5708])


def test_validate_action_rejects_gripper_rotation_key() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="gripper_rotation"):
        module.validate_action(
            {
                "section_angles": [2.1, 1.5708, 1.5, 1.5708, 1.0, 1.5708],
                "grip_command": 0.0,
                "gripper_rotation": 0.0,
            }
        )


def test_build_trial_grid_respects_max_trials() -> None:
    module = _load_module()

    trials = module.build_trial_grid(
        preclose_scales=[0.70, 0.75],
        close_advance_scales=[0.00, 0.05],
        grasper_rotations=[0.0, 0.7854],
        max_trials=6,
    )

    assert len(trials) == 6
    assert all(0.0 <= trial["final_close_scale"] <= 1.0 for trial in trials)


def test_clean_preclose_and_close_added_contact_flags() -> None:
    module = _load_module()

    result = module.evaluate_trial_outcome(
        preclose_had_contact=False,
        red_displacement_before_close=0.002,
        close_had_contact=True,
        hold_closed_had_contact=False,
        red_displacement_after_close=0.0065,
        red_displacement_before_close_baseline=0.002,
    )

    assert result["clean_preclose"] is True
    assert result["close_added_contact"] is True
    assert result["close_added_motion"] is True
    assert result["close_added_red_displacement"] == pytest.approx(0.0045)


def test_rank_trials_prefers_clean_close_then_near_preclose() -> None:
    module = _load_module()
    trials = [
        {
            "name": "weak_near",
            "close_added_contact": False,
            "preclose_had_contact": False,
            "preclose_final_distance": 0.06,
            "close_added_red_displacement": 0.001,
        },
        {
            "name": "clean_close",
            "close_added_contact": True,
            "preclose_had_contact": False,
            "preclose_final_distance": 0.11,
            "close_added_red_displacement": 0.004,
        },
        {
            "name": "contact_but_dirty",
            "close_added_contact": True,
            "preclose_had_contact": True,
            "preclose_final_distance": 0.03,
            "close_added_red_displacement": 0.010,
        },
    ]

    ranked = module.rank_trials(trials)

    assert ranked[0]["name"] == "clean_close"
    assert ranked[1]["name"] == "contact_but_dirty"


def test_contact_classification_detects_grasper_body_not_pedestal() -> None:
    module = _load_module()

    red_grasper = module.classify_contacts(
        [
            {
                "geom1": "<unnamed:41>",
                "geom2": "red_object_geom",
                "body1": "feagine_grasper_finger_0",
                "body2": "red_object",
                "distance": -0.001,
                "normal_force": 0.5,
            }
        ]
    )
    red_pedestal = module.classify_contacts(
        [
            {
                "geom1": "red_object_geom",
                "geom2": "red_pedestal_geom",
                "body1": "red_object",
                "body2": "red_pedestal",
                "distance": -0.0005,
                "normal_force": 0.8,
            }
        ]
    )

    assert red_grasper["has_red_grasper_contact"] is True
    assert red_grasper["contact_details"][0]["body_pair"] == [
        "feagine_grasper_finger_0",
        "red_object",
    ]
    assert red_pedestal["has_red_grasper_contact"] is False


def test_make_judgment_covers_all_four_paths() -> None:
    module = _load_module()

    assert module.make_judgment(
        any_red_grasper_contact=True,
        best_clean_close_trial={"close_added_contact": True, "close_advance_scale": 0.0},
    ).startswith("[OK]")
    assert module.make_judgment(
        any_red_grasper_contact=True,
        best_clean_close_trial={"close_added_contact": True, "close_advance_scale": 0.10},
    ).startswith("[WARN] close contact required")
    assert module.make_judgment(
        any_red_grasper_contact=True,
        best_clean_close_trial=None,
    ).startswith("[WARN] only approach/direct contact observed")
    assert module.make_judgment(
        any_red_grasper_contact=False,
        best_clean_close_trial=None,
    ).startswith("[FAIL]")
