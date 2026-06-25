from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "test_controlled_gripper_close_contact.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "test_controlled_gripper_close_contact",
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
                "section_angles": [2.1, 1.5708, 1.5, 1.5708, 1.0, 1.5708],
                "grip_command": 0.0,
                "gripper_rotation": 0.0,
            }
        )


def test_interpolate_section_angles_ramps_from_zero_to_target() -> None:
    module = _load_module()
    target = [2.1, 1.5708, 1.5, 1.5708, 1.0, 1.5708]

    first = module.interpolate_section_angles(target, step_index=0, total_steps=160)
    last = module.interpolate_section_angles(target, step_index=159, total_steps=160)

    assert first[0] == pytest.approx(target[0] / 160.0)
    assert first[1] == pytest.approx(target[1] / 160.0)
    assert last == pytest.approx(target)


def test_interpolate_grip_command_closes_from_zero_to_one() -> None:
    module = _load_module()

    first = module.interpolate_grip_command(step_index=0, total_steps=80)
    last = module.interpolate_grip_command(step_index=79, total_steps=80)

    assert first == pytest.approx(1.0 / 80.0)
    assert last == pytest.approx(1.0)


def test_contact_classification_detects_red_grasper_not_pedestal() -> None:
    module = _load_module()
    red_grasper_flags = module.classify_contacts(
        [
            {
                "geom1": "red_object_geom",
                "geom2": "<unnamed:41>",
                "body1": "red_object",
                "body2": "feagine_grasper_finger_left",
                "normal_force": 0.7,
            }
        ]
    )
    pedestal_flags = module.classify_contacts(
        [
            {
                "geom1": "red_object_geom",
                "geom2": "red_pedestal_geom",
                "body1": "red_object",
                "body2": "red_pedestal",
                "normal_force": 0.9,
            }
        ]
    )

    assert red_grasper_flags["has_red_grasper_contact"] is True
    assert red_grasper_flags["max_red_grasper_normal_force"] == pytest.approx(0.7)
    assert pedestal_flags["has_red_grasper_contact"] is False
    assert pedestal_flags["has_red_pedestal_contact"] is True


def test_trial_summary_aggregation_collects_phase_stats() -> None:
    module = _load_module()
    records = [
        {
            "phase": "approach_ramp",
            "tip_red_distance": 0.3,
            "red_displacement": 0.01,
            "has_red_grasper_contact": False,
            "has_red_pedestal_contact": True,
            "has_red_table_contact": False,
            "max_red_grasper_normal_force": 0.0,
            "total_red_grasper_normal_force": 0.0,
            "contact_pairs": ["red_object_geom <-> red_pedestal_geom"],
        },
        {
            "phase": "close_ramp",
            "tip_red_distance": 0.05,
            "red_displacement": 0.03,
            "has_red_grasper_contact": True,
            "has_red_pedestal_contact": True,
            "has_red_table_contact": False,
            "max_red_grasper_normal_force": 1.2,
            "total_red_grasper_normal_force": 2.4,
            "contact_pairs": ["<unnamed:41> <-> red_object_geom"],
        },
        {
            "phase": "hold_closed",
            "tip_red_distance": 0.06,
            "red_displacement": 0.025,
            "has_red_grasper_contact": True,
            "has_red_pedestal_contact": True,
            "has_red_table_contact": False,
            "max_red_grasper_normal_force": 0.8,
            "total_red_grasper_normal_force": 1.6,
            "contact_pairs": ["<unnamed:42> <-> red_object_geom"],
        },
    ]

    summary = module.summarize_trial(
        name="controlled_close",
        initial_tip_position=[0.0, 0.0, 0.6],
        final_tip_position=[0.3, 0.0, 0.35],
        initial_red_position=[0.34, 0.0, 0.33],
        final_red_position=[0.31, 0.0, 0.33],
        records=records,
    )

    assert summary["min_tip_red_distance"] == pytest.approx(0.05)
    assert summary["max_red_displacement"] == pytest.approx(0.03)
    assert summary["red_displacement_during_approach"] == pytest.approx(0.01)
    assert summary["red_displacement_during_close"] == pytest.approx(0.03)
    assert summary["has_red_grasper_contact_during_close"] is True
    assert summary["has_red_grasper_contact_during_hold_closed"] is True
    assert summary["max_red_grasper_normal_force"] == pytest.approx(1.2)


def test_comparison_logic_detects_added_close_effect() -> None:
    module = _load_module()
    comparison = module.compare_trials(
        open_only_baseline={
            "max_red_displacement": 0.01,
            "min_tip_red_distance": 0.06,
            "max_red_grasper_normal_force": 0.3,
            "has_red_grasper_contact_during_close": False,
            "has_red_grasper_contact_during_hold_closed": False,
        },
        controlled_close={
            "max_red_displacement": 0.03,
            "min_tip_red_distance": 0.04,
            "max_red_grasper_normal_force": 1.4,
            "has_red_grasper_contact_during_close": True,
            "has_red_grasper_contact_during_hold_closed": True,
        },
    )

    assert comparison["close_added_red_displacement"] == pytest.approx(0.02)
    assert comparison["close_added_contact"] is True
    assert comparison["close_max_force_minus_baseline"] == pytest.approx(1.1)
    assert comparison["close_min_distance_minus_baseline"] == pytest.approx(-0.02)


def test_make_judgment_covers_ok_warn_and_fail_paths() -> None:
    module = _load_module()

    assert module.make_judgment(
        open_only_baseline={"has_red_grasper_contact_during_approach": False},
        controlled_close={
            "has_red_grasper_contact_during_approach": False,
            "has_red_grasper_contact_during_close": True,
            "has_red_grasper_contact_during_hold_closed": True,
        },
        comparison={"close_added_contact": True, "close_added_red_displacement": 0.02, "close_max_force_minus_baseline": 0.5},
    ).startswith("[OK]")
    assert module.make_judgment(
        open_only_baseline={"has_red_grasper_contact_during_approach": True},
        controlled_close={
            "has_red_grasper_contact_during_approach": True,
            "has_red_grasper_contact_during_close": True,
            "has_red_grasper_contact_during_hold_closed": False,
        },
        comparison={"close_added_contact": False, "close_added_red_displacement": 0.001, "close_max_force_minus_baseline": 0.01},
    ).startswith("[WARN] approach produced")
    assert module.make_judgment(
        open_only_baseline={"has_red_grasper_contact_during_approach": False},
        controlled_close={
            "has_red_grasper_contact_during_approach": True,
            "has_red_grasper_contact_during_close": False,
            "has_red_grasper_contact_during_hold_closed": False,
        },
        comparison={"close_added_contact": False, "close_added_red_displacement": 0.0, "close_max_force_minus_baseline": 0.0},
    ).startswith("[WARN] close did not add")
    assert module.make_judgment(
        open_only_baseline={"has_red_grasper_contact_during_approach": False},
        controlled_close={
            "has_red_grasper_contact_during_approach": False,
            "has_red_grasper_contact_during_close": False,
            "has_red_grasper_contact_during_hold_closed": False,
        },
        comparison={"close_added_contact": False, "close_added_red_displacement": 0.0, "close_max_force_minus_baseline": 0.0},
    ).startswith("[FAIL]")
