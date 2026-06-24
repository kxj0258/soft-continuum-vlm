from __future__ import annotations

import importlib
import sys
from typing import Any


PRIMARY_PRESET_ID = "a03_type_2"
FALLBACK_PRESET_ID = "a03"
MODEL_TYPE = "mjcf"


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")


def _import_module(module_name: str) -> Any | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        _fail(f"import {module_name}: {exc}")
        return None
    _ok(f"import {module_name}")
    return module


def _check_feagine_api(feagine_mujoco: Any) -> dict[str, Any]:
    names = [
        "robot_asset_path",
        "load_mujoco_model",
        "FeagineMjcfRobot",
        "FeagineUrdfRobot",
    ]
    resolved: dict[str, Any] = {}
    for name in names:
        value = getattr(feagine_mujoco, name, None)
        if value is None:
            _fail(f"feagine_mujoco has {name}: no")
        else:
            _ok(f"feagine_mujoco has {name}: yes")
            resolved[name] = value
    return resolved


def _resolve_asset_path(robot_asset_path: Any, preset_id: str) -> Any | None:
    try:
        path = robot_asset_path(preset_id=preset_id, model_type=MODEL_TYPE)
    except Exception as exc:
        _fail(f"robot_asset_path(preset_id={preset_id!r}, model_type={MODEL_TYPE!r}): {exc}")
        return None
    _ok(f"robot_asset_path(preset_id={preset_id!r}, model_type={MODEL_TYPE!r}): {path}")
    return path


def _select_preset(robot_asset_path: Any) -> str | None:
    if _resolve_asset_path(robot_asset_path, PRIMARY_PRESET_ID) is not None:
        return PRIMARY_PRESET_ID
    _warn("a03_type_2 unavailable; trying a03 only for diagnosis.")
    if _resolve_asset_path(robot_asset_path, FALLBACK_PRESET_ID) is not None:
        return FALLBACK_PRESET_ID
    return None


def _load_model(load_mujoco_model: Any, preset_id: str) -> Any | None:
    try:
        model = load_mujoco_model(preset_id=preset_id, model_type=MODEL_TYPE)
    except Exception as exc:
        _fail(f"load_mujoco_model(preset_id={preset_id!r}, model_type={MODEL_TYPE!r}): {exc}")
        return None
    _ok(f"load_mujoco_model(preset_id={preset_id!r}, model_type={MODEL_TYPE!r})")
    return model


def _create_data(mujoco: Any, model: Any) -> Any | None:
    try:
        data = mujoco.MjData(model)
    except Exception as exc:
        _fail(f"mujoco.MjData(model): {exc}")
        return None
    _ok("mujoco.MjData(model)")
    return data


def _create_robot(robot_class: Any, model: Any, data: Any, preset_id: str) -> Any | None:
    try:
        robot = robot_class(model, data, preset_id=preset_id)
    except Exception as exc:
        _fail(f"FeagineMjcfRobot(model, data, preset_id={preset_id!r}): {exc}")
        return None
    _ok(f"FeagineMjcfRobot(model, data, preset_id={preset_id!r})")
    return robot


def _print_model_info(model: Any) -> None:
    for name in ("nq", "nv", "nu", "nbody", "ngeom", "nsite", "nactuator"):
        if hasattr(model, name):
            _ok(f"model.{name}: {getattr(model, name)}")
        else:
            _warn(f"model.{name}: unavailable")


def _print_robot_info(robot: Any) -> None:
    if hasattr(robot, "section_count"):
        _ok(f"robot.section_count: {getattr(robot, 'section_count')}")
    else:
        _warn("robot.section_count: unavailable")

    get_section_angles = getattr(robot, "get_section_angles", None)
    if callable(get_section_angles):
        try:
            section_angles = get_section_angles()
        except Exception as exc:
            _warn(f"robot.get_section_angles(): {exc}")
        else:
            _ok(f"robot.get_section_angles(): {section_angles}")
    else:
        _warn("robot.get_section_angles(): unavailable")

    for name in ("grip_command", "grasper_rotation"):
        if hasattr(robot, name):
            _ok(f"robot.{name}: {getattr(robot, name)}")
        else:
            _warn(f"robot.{name}: unavailable")


def _print_robot_method_info(robot: Any) -> None:
    for name in (
        "drive_section_angles",
        "set_grip_command",
        "drive_grasper_rotation",
        "drive_joint_targets",
        "set_segment_joint_targets",
    ):
        if callable(getattr(robot, name, None)):
            _ok(f"robot has method {name}: yes")
        else:
            _warn(f"robot has method {name}: no")


def main() -> int:
    modules = {
        "pyfeagine_sim_core": _import_module("pyfeagine_sim_core"),
        "feagine_mujoco": _import_module("feagine_mujoco"),
        "mujoco": _import_module("mujoco"),
    }
    if any(module is None for module in modules.values()):
        return 1

    feagine_mujoco = modules["feagine_mujoco"]
    mujoco = modules["mujoco"]
    api = _check_feagine_api(feagine_mujoco)

    robot_asset_path = api.get("robot_asset_path")
    if robot_asset_path is None:
        return 2
    preset_id = _select_preset(robot_asset_path)
    if preset_id is None:
        return 2

    load_mujoco_model = api.get("load_mujoco_model")
    robot_class = api.get("FeagineMjcfRobot")
    if load_mujoco_model is None or robot_class is None:
        return 2

    model = _load_model(load_mujoco_model, preset_id)
    if model is None:
        return 2
    _print_model_info(model)

    data = _create_data(mujoco, model)
    if data is None:
        return 2

    robot = _create_robot(robot_class, model, data, preset_id)
    if robot is None:
        return 2
    _print_robot_info(robot)
    _print_robot_method_info(robot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
