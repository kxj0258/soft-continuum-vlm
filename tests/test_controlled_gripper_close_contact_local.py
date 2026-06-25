from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "test_controlled_gripper_close_contact_local.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "test_controlled_gripper_close_contact_local",
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


def test_select_preclose_candidate_prefers_clean_midrange_distance() -> None:
    module = _load_module()
    candidates = [
        {
            "scale": 0.95,
            "final_tip_red_distance": 0.04,
            "has_red_grasper_contact": False,
            "max_red_displacement": 0.002,
        },
        {
            "scale": 0.90,
            "final_tip_red_distance": 0.08,
            "has_red_grasper_contact": False,
            "max_red_displacement": 0.003,
        },
        {
            "scale": 0.85,
            "final_tip_red_distance": 0.09,
            "has_red_grasper_contact": False,
            "max_red_displacement": 0.006,
        },
    ]

    selected = module.select_preclose_candidate(candidates)

    assert selected["scale"] == pytest.approx(0.90)
    assert selected["clean_preclose_found"] is True


def test_select_preclose_candidate_falls_back_to_closest_no_contact() -> None:
    module = _load_module()
    candidates = [
        {
            "scale": 0.95,
            "final_tip_red_distance": 0.03,
            "has_red_grasper_contact": False,
            "max_red_displacement": 0.010,
        },
        {
            "scale": 0.90,
            "final_tip_red_distance": 0.18,
            "has_red_grasper_contact": False,
            "max_red_displacement": 0.003,
        },
        {
            "scale": 0.85,
            "final_tip_red_distance": 0.07,
            "has_red_grasper_contact": True,
            "max_red_displacement": 0.001,
        },
    ]

    selected = module.select_preclose_candidate(candidates)

    assert selected["scale"] == pytest.approx(0.95)
    assert selected["clean_preclose_found"] is False


def test_contact_classification_uses_body_names_for_unnamed_geoms() -> None:
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
    assert red_grasper["max_red_grasper_normal_force"] == pytest.approx(0.5)
    assert red_pedestal["has_red_grasper_contact"] is False
    assert red_pedestal["has_red_pedestal_contact"] is True


def test_build_comparison_uses_direct_and_preclose_metrics() -> None:
    module = _load_module()

    comparison = module.build_comparison(
        direct_open_repro={
            "max_red_displacement": 0.020,
            "max_red_grasper_normal_force": 0.40,
            "has_red_grasper_contact": True,
        },
        direct_close_repro={
            "max_red_displacement": 0.027,
            "max_red_grasper_normal_force": 0.90,
            "has_red_grasper_contact": True,
        },
        preclose_scaled_close={
            "close_added_contact": True,
            "close_added_red_displacement": 0.004,
        },
    )

    assert comparison["direct_close_minus_open_red_displacement"] == pytest.approx(0.007)
    assert comparison["direct_close_minus_open_max_force"] == pytest.approx(0.5)
    assert comparison["preclose_close_added_contact"] is True
    assert comparison["preclose_close_added_red_displacement"] == pytest.approx(0.004)


def test_make_judgment_covers_all_result_paths() -> None:
    module = _load_module()

    assert module.make_judgment(
        direct_open_repro={"has_red_grasper_contact": False},
        direct_close_repro={"has_red_grasper_contact": False},
        preclose_scaled_close={
            "preclose_had_contact": False,
            "has_red_grasper_contact_during_close": True,
            "has_red_grasper_contact_during_hold_closed": False,
            "clean_preclose_found": True,
        },
    ).startswith("[OK]")
    assert module.make_judgment(
        direct_open_repro={"has_red_grasper_contact": True},
        direct_close_repro={"has_red_grasper_contact": False},
        preclose_scaled_close={
            "preclose_had_contact": False,
            "has_red_grasper_contact_during_close": False,
            "has_red_grasper_contact_during_hold_closed": False,
            "clean_preclose_found": True,
        },
    ).startswith("[WARN] contact reproduced")
    assert module.make_judgment(
        direct_open_repro={"has_red_grasper_contact": False},
        direct_close_repro={"has_red_grasper_contact": True},
        preclose_scaled_close={
            "preclose_had_contact": False,
            "has_red_grasper_contact_during_close": False,
            "has_red_grasper_contact_during_hold_closed": False,
            "clean_preclose_found": False,
        },
    ).startswith("[WARN] direct contact reproduced")
    assert module.make_judgment(
        direct_open_repro={"has_red_grasper_contact": False},
        direct_close_repro={"has_red_grasper_contact": False},
        preclose_scaled_close={
            "preclose_had_contact": False,
            "has_red_grasper_contact_during_close": False,
            "has_red_grasper_contact_during_hold_closed": False,
            "clean_preclose_found": False,
        },
    ).startswith("[FAIL]")
