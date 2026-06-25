from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


ALLOWED_ACTION_KEYS = {"section_angles", "grip_command", "grasper_rotation"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the Feagine reachable workspace under PCC commands."
    )
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument(
        "--steps-per-command",
        type=int,
        default=80,
        help="Simulation steps to run for each candidate command.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=50,
        help="Passive settling steps before each command.",
    )
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    return parser.parse_args()


def build_commands() -> list[dict[str, Any]]:
    bend_patterns = [
        [0.0, 0.0, 0.0],
        [0.25, 0.15, 0.10],
        [0.50, 0.35, 0.20],
        [0.80, 0.55, 0.35],
        [1.10, 0.80, 0.50],
        [1.40, 1.00, 0.70],
    ]
    directions = [
        -3.1416,
        -2.3562,
        -1.5708,
        -0.7854,
        -0.32,
        0.0,
        0.7854,
        1.5708,
        2.3562,
        3.1416,
    ]
    mixed_commands = [
        [0.8, 0.0, 0.5, 1.5708, 0.3, 3.1416],
        [0.8, -0.32, 0.5, -0.32, 0.3, -0.32],
        [1.2, -0.32, 0.8, -0.32, 0.4, -0.32],
        [1.2, 0.0, 0.8, -0.7854, 0.4, -1.5708],
        [1.2, 1.5708, 0.8, 0.7854, 0.4, 0.0],
    ]

    commands: list[dict[str, Any]] = []
    for pattern_index, pattern in enumerate(bend_patterns):
        for direction_index, direction in enumerate(directions):
            commands.append(
                {
                    "name": f"pattern_{pattern_index:02d}_dir_{direction_index:02d}",
                    "section_angles": [pattern[0], direction, pattern[1], direction, pattern[2], direction],
                    "grip_command": 0.0,
                    "grasper_rotation": 0.0,
                }
            )

    for mixed_index, section_angles in enumerate(mixed_commands):
        commands.append(
            {
                "name": f"mixed_{mixed_index:02d}",
                "section_angles": [float(value) for value in section_angles],
                "grip_command": 0.0,
                "grasper_rotation": 0.0,
            }
        )

    return commands


def _as_float_sequence(value: Any, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{label} must be a numeric sequence.")
    return [float(item) for item in value]


def validate_action(action: dict[str, Any]) -> None:
    unsupported = sorted(set(action) - ALLOWED_ACTION_KEYS)
    if unsupported:
        raise ValueError(f"unsupported action keys: {unsupported}")
    if "gripper_rotation" in action:
        raise ValueError("unsupported action keys: ['gripper_rotation']")

    section_angles = _as_float_sequence(action["section_angles"], "section_angles")
    if len(section_angles) != 6:
        raise ValueError("section_angles must contain 6 values.")
    float(action["grip_command"])
    float(action["grasper_rotation"])


def _distance(position_a: Sequence[float], position_b: Sequence[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(position_a, dtype=np.float64) - np.asarray(position_b, dtype=np.float64)
        )
    )


def _distance_reduction_ratio(initial_distance: float, min_distance: float) -> float:
    if initial_distance <= 1e-12:
        return 0.0
    return float(max(0.0, (float(initial_distance) - float(min_distance)) / float(initial_distance)))


def _name_matches(value: str, needles: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(needle in lowered for needle in needles)


def _load_tabletop_scene(scene_path: str) -> tuple[Any, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    return model, data


def _create_feagine_wrapper(model: Any, data: Any) -> Any:
    import feagine_mujoco

    robot_class = getattr(feagine_mujoco, "FeagineMjcfRobot", None)
    if robot_class is None:
        raise RuntimeError("feagine_mujoco.FeagineMjcfRobot is unavailable.")
    return robot_class(model, data, preset_id="a03_type_2")


def _apply_feagine_action(robot: Any, action: dict[str, Any]) -> None:
    validate_action(action)

    drive_section_angles = getattr(robot, "drive_section_angles", None)
    section_count = getattr(robot, "section_count", None)
    if not callable(drive_section_angles) or section_count is None:
        raise RuntimeError(
            "Missing Feagine wrapper section angle interface: drive_section_angles/section_count."
        )
    section_angles = _as_float_sequence(action["section_angles"], "section_angles")
    expected = int(section_count) * 2
    if len(section_angles) != expected:
        raise ValueError(f"section_angles must contain {expected} values, got {len(section_angles)}.")
    drive_section_angles(section_angles)

    set_grip_command = getattr(robot, "set_grip_command", None)
    if not callable(set_grip_command):
        raise RuntimeError("Missing Feagine wrapper gripper interface: set_grip_command.")
    set_grip_command(float(action["grip_command"]))

    drive_grasper_rotation = getattr(robot, "drive_grasper_rotation", None)
    if not callable(drive_grasper_rotation):
        raise RuntimeError("Missing Feagine wrapper rotation interface: drive_grasper_rotation.")
    drive_grasper_rotation(float(action["grasper_rotation"]))


def _body_id(mujoco: Any, model: Any, body_name: str) -> int | None:
    try:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    except Exception:
        return None
    return body_id if body_id >= 0 else None


def _body_position(mujoco: Any, model: Any, data: Any, body_name: str) -> list[float] | None:
    body_id = _body_id(mujoco, model, body_name)
    if body_id is None:
        return None
    position = data.xpos[body_id]
    return [float(position[index]) for index in range(3)]


def _geom_name(model: Any, geom_id: int) -> str:
    name = ""
    try:
        name = str(model.geom(geom_id).name or "")
    except Exception:
        name = ""
    return name if name else f"<unnamed:{int(geom_id)}>"


def _body_name(model: Any, body_id: int) -> str:
    name = ""
    try:
        name = str(model.body(body_id).name or "")
    except Exception:
        name = ""
    return name if name else f"<unnamed:{int(body_id)}>"


def _contact_snapshot(model: Any, data: Any) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        body1_id = int(model.geom_bodyid[geom1_id])
        body2_id = int(model.geom_bodyid[geom2_id])
        contacts.append(
            {
                "geom1": _geom_name(model, geom1_id),
                "geom2": _geom_name(model, geom2_id),
                "body1": _body_name(model, body1_id),
                "body2": _body_name(model, body2_id),
            }
        )
    return contacts


def _contact_pair_strings(contacts: Sequence[dict[str, Any]]) -> list[str]:
    pairs = {
        " <-> ".join(sorted([str(contact["geom1"]), str(contact["geom2"])]))
        for contact in contacts
    }
    return sorted(pairs)


def _scene_entity_positions(mujoco: Any, model: Any, data: Any) -> dict[str, list[float] | None]:
    return {
        "feagine_grasper_tip": _body_position(mujoco, model, data, "feagine_grasper_tip"),
        "red_object": _body_position(mujoco, model, data, "red_object"),
        "blue_object": _body_position(mujoco, model, data, "blue_object"),
        "black_obstacle": _body_position(mujoco, model, data, "black_obstacle"),
        "target_pad": _body_position(mujoco, model, data, "target_pad"),
    }


def _summarize_tip_positions(tip_positions: Sequence[Sequence[float]]) -> dict[str, float]:
    array = np.asarray(tip_positions, dtype=np.float64)
    return {
        "x_min": float(np.min(array[:, 0])),
        "x_max": float(np.max(array[:, 0])),
        "y_min": float(np.min(array[:, 1])),
        "y_max": float(np.max(array[:, 1])),
        "z_min": float(np.min(array[:, 2])),
        "z_max": float(np.max(array[:, 2])),
    }


def summarize_workspace(command_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not command_summaries:
        return {
            "num_commands": 0,
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
            "z_max": 0.0,
            "center": [0.0, 0.0, 0.0],
            "extent": [0.0, 0.0, 0.0],
        }

    x_min = min(float(command["min_tip_position"][0]) for command in command_summaries)
    x_max = max(float(command["max_tip_position"][0]) for command in command_summaries)
    y_min = min(float(command["min_tip_position"][1]) for command in command_summaries)
    y_max = max(float(command["max_tip_position"][1]) for command in command_summaries)
    z_min = min(float(command["min_tip_position"][2]) for command in command_summaries)
    z_max = max(float(command["max_tip_position"][2]) for command in command_summaries)
    center = [
        float((x_min + x_max) / 2.0),
        float((y_min + y_max) / 2.0),
        float((z_min + z_max) / 2.0),
    ]
    extent = [float(x_max - x_min), float(y_max - y_min), float(z_max - z_min)]

    return {
        "num_commands": int(len(command_summaries)),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "center": center,
        "extent": extent,
    }


def red_inside_xy_bbox(red_position: Sequence[float], workspace: dict[str, Any]) -> bool:
    return bool(
        float(workspace["x_min"]) <= float(red_position[0]) <= float(workspace["x_max"])
        and float(workspace["y_min"]) <= float(red_position[1]) <= float(workspace["y_max"])
    )


def red_inside_xyz_bbox(red_position: Sequence[float], workspace: dict[str, Any]) -> bool:
    return bool(
        red_inside_xy_bbox(red_position, workspace)
        and float(workspace["z_min"]) <= float(red_position[2]) <= float(workspace["z_max"])
    )


def choose_best_command(command_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not command_summaries:
        raise ValueError("No command summaries available.")

    best = min(
        command_summaries,
        key=lambda item: (
            float(item["min_tip_red_distance"]),
            float(item["final_tip_red_distance"]),
            float(
                item.get(
                    "distance_reduction_ratio",
                    _distance_reduction_ratio(
                        float(item["initial_tip_red_distance"]),
                        float(item["min_tip_red_distance"]),
                    ),
                )
            ),
        ),
    )
    best_final_tip_position = best.get(
        "final_tip_position",
        best.get("max_tip_position", best.get("min_tip_position", [0.0, 0.0, 0.0])),
    )
    return {
        "name": best["name"],
        "initial_tip_red_distance": float(best["initial_tip_red_distance"]),
        "min_tip_red_distance": float(best["min_tip_red_distance"]),
        "final_tip_red_distance": float(best["final_tip_red_distance"]),
        "distance_reduction_ratio": float(
            best.get(
                "distance_reduction_ratio",
                _distance_reduction_ratio(
                    float(best["initial_tip_red_distance"]),
                    float(best["min_tip_red_distance"]),
                ),
            )
        ),
        "best_final_tip_position": [float(value) for value in best_final_tip_position],
    }


def suggest_reachable_red_position(
    *, red_position: Sequence[float], best_command: dict[str, Any]
) -> dict[str, Any]:
    suggested_position = [float(value) for value in best_command["best_final_tip_position"]]
    reason = (
        "Best sampled reachable tip pose is the closest point to red_object under the sampled "
        "PCC commands."
    )
    return {"position": suggested_position, "reason": reason}


def make_judgment(
    *,
    red_inside_xyz_bbox: bool,
    red_inside_xy_bbox: bool,
    best_min_tip_red_distance: float,
) -> str:
    if red_inside_xyz_bbox and float(best_min_tip_red_distance) < 0.10:
        return "[OK] red_object is within sampled reachable workspace"
    if red_inside_xy_bbox and not red_inside_xyz_bbox:
        return "[WARN] red_object is near sampled xy workspace but z is not reachable"
    return "[FAIL] red_object is outside sampled reachable workspace"


def _command_tip_summary(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    command: dict[str, Any],
    steps_per_command: int,
    settle_steps: int,
) -> dict[str, Any]:
    mujoco.mj_resetData(model, data)
    try:
        robot = _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc
    mujoco.mj_forward(model, data)

    for _ in range(int(max(0, settle_steps))):
        mujoco.mj_step(model, data)

    initial_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    red_position = _body_position(mujoco, model, data, "red_object")
    if initial_tip_position is None:
        raise RuntimeError("Missing feagine_grasper_tip body pose.")

    command_payload = {
        "section_angles": [float(value) for value in command["section_angles"]],
        "grip_command": float(command["grip_command"]),
        "grasper_rotation": float(command["grasper_rotation"]),
    }

    sampled_tip_positions: list[list[float]] = [list(initial_tip_position)]
    contacts_seen: set[str] = set(_contact_pair_strings(_contact_snapshot(model, data)))
    min_tip_red_distance = (
        _distance(initial_tip_position, red_position) if red_position is not None else float("inf")
    )
    final_tip_position = list(initial_tip_position)
    final_red_position = list(red_position) if red_position is not None else None
    any_contact = int(data.ncon) > 0
    red_object_moved = False
    initial_red_position = list(red_position) if red_position is not None else None

    for _ in range(int(max(0, steps_per_command))):
        _apply_feagine_action(robot, command_payload)
        mujoco.mj_step(model, data)

        current_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
        current_red_position = _body_position(mujoco, model, data, "red_object")
        if current_tip_position is None:
            raise RuntimeError("Missing feagine_grasper_tip body pose.")

        sampled_tip_positions.append(list(current_tip_position))
        final_tip_position = list(current_tip_position)
        final_red_position = list(current_red_position) if current_red_position is not None else None
        if current_red_position is not None and initial_red_position is not None:
            red_object_moved = red_object_moved or _distance(initial_red_position, current_red_position) > 1e-4
            min_tip_red_distance = min(
                min_tip_red_distance,
                _distance(current_tip_position, current_red_position),
            )

        current_contacts = _contact_snapshot(model, data)
        any_contact = any_contact or int(data.ncon) > 0
        contacts_seen.update(_contact_pair_strings(current_contacts))

    tip_extents = _summarize_tip_positions(sampled_tip_positions)
    final_tip_red_distance = (
        _distance(final_tip_position, final_red_position)
        if final_red_position is not None
        else float("inf")
    )

    return {
        "name": command["name"],
        "action": {
            "section_angles": [float(value) for value in command["section_angles"]],
            "grip_command": float(command["grip_command"]),
            "grasper_rotation": float(command["grasper_rotation"]),
        },
        "initial_tip_position": [float(value) for value in initial_tip_position],
        "final_tip_position": [float(value) for value in final_tip_position],
        "initial_red_position": [float(value) for value in initial_red_position]
        if initial_red_position is not None
        else None,
        "final_red_position": [float(value) for value in final_red_position]
        if final_red_position is not None
        else None,
        "initial_tip_red_distance": float(
            _distance(initial_tip_position, initial_red_position)
            if initial_red_position is not None
            else float("inf")
        ),
        "min_tip_red_distance": float(min_tip_red_distance),
        "final_tip_red_distance": float(final_tip_red_distance),
        "distance_reduction_ratio": float(
            _distance_reduction_ratio(
                _distance(initial_tip_position, initial_red_position)
                if initial_red_position is not None
                else float("inf"),
                float(min_tip_red_distance),
            )
        ),
        "min_tip_position": [
            float(np.min(np.asarray(sampled_tip_positions, dtype=np.float64)[:, 0])),
            float(np.min(np.asarray(sampled_tip_positions, dtype=np.float64)[:, 1])),
            float(np.min(np.asarray(sampled_tip_positions, dtype=np.float64)[:, 2])),
        ],
        "max_tip_position": [
            float(np.max(np.asarray(sampled_tip_positions, dtype=np.float64)[:, 0])),
            float(np.max(np.asarray(sampled_tip_positions, dtype=np.float64)[:, 1])),
            float(np.max(np.asarray(sampled_tip_positions, dtype=np.float64)[:, 2])),
        ],
        "tip_extents": tip_extents,
        "any_contact": bool(any_contact),
        "red_object_moved": bool(red_object_moved),
        "contact_pairs_seen": sorted(pair for pair in contacts_seen if pair),
    }


def scan_reachability(
    scene_path: str | Path,
    *,
    steps_per_command: int,
    settle_steps: int,
) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model, data = _load_tabletop_scene(str(resolved_scene))
    mujoco.mj_forward(model, data)

    tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    if tip_position is None:
        raise ValueError("Missing required body: feagine_grasper_tip")

    try:
        _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(int(max(0, settle_steps))):
        mujoco.mj_step(model, data)

    tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    if tip_position is None:
        raise ValueError("Missing required body: feagine_grasper_tip")

    scene_entities = _scene_entity_positions(mujoco, model, data)
    commands = []
    for command in build_commands():
        summary = _command_tip_summary(
            mujoco,
            model,
            data,
            command=command,
            steps_per_command=steps_per_command,
            settle_steps=settle_steps,
        )
        commands.append(summary)

    workspace = summarize_workspace(commands)
    best_command = choose_best_command(commands)

    red_position = scene_entities["red_object"]
    if red_position is None:
        raise RuntimeError("Missing required body: red_object")

    red_inside_xy = red_inside_xy_bbox(red_position, workspace)
    red_inside_xyz = red_inside_xyz_bbox(red_position, workspace)
    suggestion = suggest_reachable_red_position(
        red_position=red_position,
        best_command=best_command,
    )
    red_object_analysis = {
        "red_position": [float(value) for value in red_position],
        "initial_tip_position": [float(value) for value in tip_position],
        "initial_tip_red_distance": float(_distance(tip_position, red_position)),
        "best_command_name": best_command["name"],
        "best_min_tip_red_distance": float(best_command["min_tip_red_distance"]),
        "best_final_tip_position": [float(value) for value in best_command["best_final_tip_position"]],
        "distance_reduction_ratio": float(best_command["distance_reduction_ratio"]),
        "red_z_minus_workspace_z_min": float(float(red_position[2]) - float(workspace["z_min"])),
        "red_z_minus_workspace_z_max": float(float(red_position[2]) - float(workspace["z_max"])),
        "red_inside_xyz_bbox": bool(red_inside_xyz),
        "red_inside_xy_bbox": bool(red_inside_xy),
        "suggested_reachable_red_position": suggestion["position"],
        "suggested_reason": suggestion["reason"],
    }
    judgment = make_judgment(
        red_inside_xyz_bbox=red_inside_xyz,
        red_inside_xy_bbox=red_inside_xy,
        best_min_tip_red_distance=float(best_command["min_tip_red_distance"]),
    )

    return {
        "scene": str(resolved_scene),
        "steps_per_command": int(max(0, steps_per_command)),
        "settle_steps": int(max(0, settle_steps)),
        "scene_entities": scene_entities,
        "workspace": workspace,
        "red_object_analysis": red_object_analysis,
        "commands": commands,
        "judgment": judgment,
        "notes": [
            "This script samples reachable tip positions under PCC commands.",
            "It does not move red_object or implement grasping.",
        ],
    }


def write_scan_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        report = scan_reachability(
            scene_path,
            steps_per_command=args.steps_per_command,
            settle_steps=args.settle_steps,
        )
    except ValueError as exc:
        if "feagine_grasper_tip" in str(exc):
            print("[FAIL] missing required body: feagine_grasper_tip")
            return 2
        print(f"[FAIL] reachability scan failed: {exc}")
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if "wrapper creation failed" in message:
            print("[FAIL] Failed to create Feagine wrapper for tabletop scene.")
            print(f"[DETAIL] {message.removeprefix('wrapper creation failed: ')}")
            return 3
        if "unsupported action keys" in message or "section_angles" in message or "Missing Feagine wrapper" in message:
            print("[FAIL] Could not apply Feagine action to tabletop scene.")
            print(f"[DETAIL] {message}")
            return 4
        print(f"[FAIL] reachability scan failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] reachability scan failed: {exc}")
        return 1

    print(f"[INFO] loaded scene: {scene_path}")
    for index, command in enumerate(report["commands"]):
        print(
            f"[CMD {index:03d}] name={command['name']} "
            f"final_tip={command['final_tip_position']} "
            f"min_tip_red_distance={command['min_tip_red_distance']}"
        )

    workspace = report["workspace"]
    print(
        f"[WORKSPACE] x=[{workspace['x_min']},{workspace['x_max']}] "
        f"y=[{workspace['y_min']},{workspace['y_max']}] "
        f"z=[{workspace['z_min']},{workspace['z_max']}]"
    )
    red_object_analysis = report["red_object_analysis"]
    print(f"[RED] position={red_object_analysis['red_position']}")
    print(
        f"[RED] best_min_tip_red_distance={red_object_analysis['best_min_tip_red_distance']}"
    )
    print(
        "[RED] suggested_reachable_red_position="
        f"{red_object_analysis['suggested_reachable_red_position']}"
    )
    print(f"[RESULT] judgment={report['judgment']}")

    try:
        saved_path = write_scan_report(report, output_path)
    except Exception as exc:
        print(f"[FAIL] failed to write diagnostics JSON: {exc}")
        return 5

    print(f"[OK] wrote {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
