from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "scan_feagine_reachability.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "scan_feagine_reachability",
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
    assert hasattr(module, "summarize_workspace")
    assert hasattr(module, "make_judgment")


def test_validate_action_rejects_gripper_rotation() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="gripper_rotation"):
        module.validate_action(
            {
                "section_angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "grip_command": 0.0,
                "grasper_rotation": 0.0,
                "gripper_rotation": 0.1,
            }
        )


def test_workspace_summary_uses_command_extents() -> None:
    module = _load_module()
    summary = module.summarize_workspace(
        [
            {
                "min_tip_position": [-1.0, 0.0, 0.2],
                "max_tip_position": [0.2, 1.0, 0.8],
            },
            {
                "min_tip_position": [-0.5, -1.2, 0.1],
                "max_tip_position": [1.4, 0.8, 1.2],
            },
        ]
    )

    assert summary["x_min"] == pytest.approx(-1.0)
    assert summary["x_max"] == pytest.approx(1.4)
    assert summary["y_min"] == pytest.approx(-1.2)
    assert summary["y_max"] == pytest.approx(1.0)
    assert summary["z_min"] == pytest.approx(0.1)
    assert summary["z_max"] == pytest.approx(1.2)
    assert summary["num_commands"] == 2


def test_red_object_bbox_checks() -> None:
    module = _load_module()
    workspace = {"x_min": 0.0, "x_max": 0.3, "y_min": -0.2, "y_max": 0.2, "z_min": 0.0, "z_max": 0.7}
    red_position = [0.24, -0.08, 0.025]

    assert module.red_inside_xy_bbox(red_position, workspace) is True
    assert module.red_inside_xyz_bbox(red_position, workspace) is True


def test_choose_best_command_prefers_smallest_distance() -> None:
    module = _load_module()
    summaries = [
        {
            "name": "a",
            "min_tip_red_distance": 0.5,
            "final_tip_red_distance": 0.6,
            "initial_tip_red_distance": 1.0,
            "min_tip_position": [0.0, 0.0, 0.0],
            "max_tip_position": [0.0, 0.0, 0.0],
        },
        {
            "name": "b",
            "min_tip_red_distance": 0.2,
            "final_tip_red_distance": 0.3,
            "initial_tip_red_distance": 1.0,
            "min_tip_position": [0.0, 0.0, 0.0],
            "max_tip_position": [0.0, 0.0, 0.0],
        },
    ]

    best = module.choose_best_command(summaries)

    assert best["name"] == "b"
    assert best["min_tip_red_distance"] == pytest.approx(0.2)


def test_suggest_reachable_red_position_returns_best_tip_position() -> None:
    module = _load_module()
    suggestion = module.suggest_reachable_red_position(
        red_position=[0.24, -0.08, 0.025],
        best_command={
            "name": "bend_y_plus_light",
            "best_final_tip_position": [0.15, -0.04, 0.03],
            "best_min_tip_red_distance": 0.12,
        },
    )

    assert suggestion["position"] == [0.15, -0.04, 0.03]
    assert "reachable" in suggestion["reason"].lower()


def test_make_judgment_uses_workspace_and_distance() -> None:
    module = _load_module()

    assert module.make_judgment(
        red_inside_xyz_bbox=True,
        red_inside_xy_bbox=True,
        best_min_tip_red_distance=0.09,
    ).startswith("[OK]")

    assert module.make_judgment(
        red_inside_xyz_bbox=False,
        red_inside_xy_bbox=True,
        best_min_tip_red_distance=0.12,
    ).startswith("[WARN]")

    assert module.make_judgment(
        red_inside_xyz_bbox=False,
        red_inside_xy_bbox=False,
        best_min_tip_red_distance=0.3,
    ).startswith("[FAIL]")
