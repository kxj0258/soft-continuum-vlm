from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ALLOWED_ACTION_KEYS = {"section_angles", "grip_command", "grasper_rotation"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Feagine actions to see whether the grasper approaches red_object."
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
    parser.add_argument(
        "--include-strong",
        action="store_true",
        default=True,
        help="Include stronger bend commands. Enabled by default.",
    )
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    return parser.parse_args()


def build_commands(*, include_strong: bool = True) -> list[dict[str, Any]]:
    commands = [
        {
            "name": "straight_open",
            "section_angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        {
            "name": "bend_y_plus_light",
            "section_angles": [0.35, 1.5708, 0.25, 1.5708, 0.15, 1.5708],
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        {
            "name": "bend_y_minus_light",
            "section_angles": [0.35, -1.5708, 0.25, -1.5708, 0.15, -1.5708],
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        {
            "name": "bend_x_plus_light",
            "section_angles": [0.35, 0.0, 0.25, 0.0, 0.15, 0.0],
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        {
            "name": "bend_x_minus_light",
            "section_angles": [0.35, 3.1416, 0.25, 3.1416, 0.15, 3.1416],
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        {
            "name": "bend_diag_to_red",
            "section_angles": [0.6, -0.32, 0.45, -0.32, 0.30, -0.32],
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
    ]

    if include_strong:
        commands.extend(
            [
                {
                    "name": "bend_diag_to_red_strong",
                    "section_angles": [1.0, -0.32, 0.75, -0.32, 0.50, -0.32],
                    "grip_command": 0.0,
                    "grasper_rotation": 0.0,
                },
                {
                    "name": "bend_diag_to_red_close",
                    "section_angles": [1.0, -0.32, 0.75, -0.32, 0.50, -0.32],
                    "grip_command": 1.0,
                    "grasper_rotation": 0.0,
                },
            ]
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
    return float(np.linalg.norm(np.asarray(position_a, dtype=np.float64) - np.asarray(position_b, dtype=np.float64)))


def _distance_reduction_ratio(initial_distance: float, min_distance: float) -> float:
    if initial_distance <= 1e-12:
        return 0.0
    return float(max(0.0, (float(initial_distance) - float(min_distance)) / float(initial_distance)))


def _name_matches(value: str, needles: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(needle in lowered for needle in needles)


def contact_flags(contacts: Sequence[dict[str, Any]]) -> dict[str, bool]:
    has_red_object_contact = False
    has_red_grasper_contact = False
    has_tip_contact = False

    for contact in contacts:
        labels = [
            str(contact.get("geom1", "")),
            str(contact.get("geom2", "")),
            str(contact.get("body1", "")),
            str(contact.get("body2", "")),
        ]
        has_red = any(_name_matches(label, ("red_object",)) for label in labels)
        has_grasper = any(_name_matches(label, ("grasper", "finger", "feagine_grasper")) for label in labels)
        has_tip = any(_name_matches(label, ("tip", "feagine_grasper_tip")) for label in labels)

        has_red_object_contact = has_red_object_contact or has_red
        has_red_grasper_contact = has_red_grasper_contact or (has_red and has_grasper)
        has_tip_contact = has_tip_contact or has_tip

    return {
        "has_red_object_contact": bool(has_red_object_contact),
        "has_red_grasper_contact": bool(has_red_grasper_contact),
        "has_tip_contact": bool(has_tip_contact),
    }


def choose_best_command(command_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not command_summaries:
        raise ValueError("No command summaries available.")
    best = min(
        command_summaries,
        key=lambda item: (
            float(item["min_tip_red_distance"]),
            0 if bool(item.get("has_red_grasper_contact")) else 1,
            float(item["final_tip_red_distance"]),
        ),
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
        "has_red_grasper_contact": bool(best["has_red_grasper_contact"]),
    }


def make_judgment(
    *,
    any_red_grasper_contact: bool,
    best_initial_tip_red_distance: float,
    best_min_tip_red_distance: float,
) -> str:
    if any_red_grasper_contact:
        return "[OK] red-grasper contact observed"

    ratio = _distance_reduction_ratio(best_initial_tip_red_distance, best_min_tip_red_distance)
    if ratio >= 0.2:
        return "[WARN] grasper approached red_object but no contact"
    return "[FAIL] grasper did not approach red_object"


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
        raise ValueError(
            f"section_angles must contain {expected} values, got {len(section_angles)}."
        )
    drive_section_angles(section_angles)

    set_grip_command = getattr(robot, "set_grip_command", None)
    if not callable(set_grip_command):
        raise RuntimeError("Missing Feagine wrapper gripper interface: set_grip_command.")
    set_grip_command(float(action["grip_command"]))

    drive_grasper_rotation = getattr(robot, "drive_grasper_rotation", None)
    if not callable(drive_grasper_rotation):
        raise RuntimeError(
            "Missing Feagine wrapper rotation interface: drive_grasper_rotation."
        )
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
    return [float(value) for value in data.xpos[body_id][:3]]


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


def _scan_command(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    command: dict[str, Any],
    steps_per_command: int,
    settle_steps: int,
) -> dict[str, Any]:
    mujoco.mj_resetData(model, data)
    robot = _create_feagine_wrapper(model, data)
    mujoco.mj_forward(model, data)

    for _ in range(int(max(0, settle_steps))):
        mujoco.mj_step(model, data)

    initial_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    initial_red_position = _body_position(mujoco, model, data, "red_object")
    if initial_tip_position is None or initial_red_position is None:
        raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")

    initial_distance = _distance(initial_tip_position, initial_red_position)
    min_distance = initial_distance
    min_distance_step = 0
    max_contacts = int(data.ncon)
    pairs_seen = set(_contact_pair_strings(_contact_snapshot(model, data)))
    saw_red_grasper_contact = False
    saw_tip_contact = False
    saw_red_object_contact = False

    final_tip_position = list(initial_tip_position)
    final_red_position = list(initial_red_position)
    action_payload = {
        "section_angles": list(command["section_angles"]),
        "grip_command": float(command["grip_command"]),
        "grasper_rotation": float(command["grasper_rotation"]),
    }

    for step_index in range(int(max(0, steps_per_command))):
        try:
            _apply_feagine_action(robot, action_payload)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        mujoco.mj_step(model, data)

        final_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
        final_red_position = _body_position(mujoco, model, data, "red_object")
        if final_tip_position is None or final_red_position is None:
            raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")

        distance_now = _distance(final_tip_position, final_red_position)
        if distance_now < min_distance:
            min_distance = distance_now
            min_distance_step = step_index + 1

        contacts = _contact_snapshot(model, data)
        max_contacts = max(max_contacts, int(data.ncon))
        pairs_seen.update(_contact_pair_strings(contacts))
        flags = contact_flags(contacts)
        saw_red_grasper_contact = saw_red_grasper_contact or flags["has_red_grasper_contact"]
        saw_tip_contact = saw_tip_contact or flags["has_tip_contact"]
        saw_red_object_contact = saw_red_object_contact or flags["has_red_object_contact"]

    red_object_displacement = _distance(initial_red_position, final_red_position)
    final_distance = _distance(final_tip_position, final_red_position)

    return {
        "name": command["name"],
        "action": {
            "section_angles": [float(value) for value in command["section_angles"]],
            "grip_command": float(command["grip_command"]),
            "grasper_rotation": float(command["grasper_rotation"]),
        },
        "initial_tip_position": [float(value) for value in initial_tip_position],
        "final_tip_position": [float(value) for value in final_tip_position],
        "initial_red_position": [float(value) for value in initial_red_position],
        "final_red_position": [float(value) for value in final_red_position],
        "initial_tip_red_distance": float(initial_distance),
        "min_tip_red_distance": float(min_distance),
        "final_tip_red_distance": float(final_distance),
        "distance_reduction_ratio": _distance_reduction_ratio(initial_distance, min_distance),
        "min_distance_step": int(min_distance_step),
        "red_object_displacement": float(red_object_displacement),
        "has_red_grasper_contact": bool(saw_red_grasper_contact),
        "has_tip_contact": bool(saw_tip_contact),
        "has_red_object_contact": bool(saw_red_object_contact),
        "num_contacts_max": int(max_contacts),
        "contact_pairs_seen": sorted(pair for pair in pairs_seen if pair),
    }


def scan_commands(
    scene_path: str | Path,
    *,
    steps_per_command: int,
    settle_steps: int,
    include_strong: bool,
) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model, data = _load_tabletop_scene(str(resolved_scene))

    try:
        _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc

    command_summaries: list[dict[str, Any]] = []
    for command in build_commands(include_strong=include_strong):
        summary = _scan_command(
            mujoco,
            model,
            data,
            command=command,
            steps_per_command=steps_per_command,
            settle_steps=settle_steps,
        )
        command_summaries.append(summary)

    best_command = choose_best_command(command_summaries)
    any_red_grasper_contact = any(
        bool(command["has_red_grasper_contact"]) for command in command_summaries
    )
    any_tip_contact = any(bool(command["has_tip_contact"]) for command in command_summaries)
    max_red_displacement = max(
        float(command["red_object_displacement"]) for command in command_summaries
    )
    any_red_object_moved = bool(max_red_displacement > 1e-4)
    judgment = make_judgment(
        any_red_grasper_contact=any_red_grasper_contact,
        best_initial_tip_red_distance=float(best_command["initial_tip_red_distance"]),
        best_min_tip_red_distance=float(best_command["min_tip_red_distance"]),
    )

    return {
        "scene": str(resolved_scene),
        "steps_per_command": int(max(0, steps_per_command)),
        "settle_steps": int(max(0, settle_steps)),
        "commands": command_summaries,
        "best_command": best_command,
        "flags": {
            "any_red_grasper_contact": any_red_grasper_contact,
            "any_tip_contact": any_tip_contact,
            "any_red_object_moved": any_red_object_moved,
        },
        "max_red_object_displacement": float(max_red_displacement),
        "judgment": judgment,
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
        report = scan_commands(
            scene_path,
            steps_per_command=args.steps_per_command,
            settle_steps=args.settle_steps,
            include_strong=bool(args.include_strong),
        )
    except RuntimeError as exc:
        message = str(exc)
        if "wrapper creation failed" in message:
            print("[FAIL] Failed to create Feagine wrapper for tabletop scene.")
            print(f"[DETAIL] {message.removeprefix('wrapper creation failed: ')}")
            return 2
        if "unsupported action keys" in message or "Missing Feagine wrapper" in message or "section_angles" in message:
            print("[FAIL] Could not apply Feagine action to tabletop scene.")
            print(f"[DETAIL] {message}")
            return 3
        print(f"[FAIL] scan failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] scan failed: {exc}")
        return 1

    for command in report["commands"]:
        print(f"[CMD] name={command['name']}")
        print(
            f"[RESULT] name={command['name']} "
            f"initial_dist={command['initial_tip_red_distance']} "
            f"min_dist={command['min_tip_red_distance']} "
            f"final_dist={command['final_tip_red_distance']} "
            f"red_grasper_contact={command['has_red_grasper_contact']} "
            f"red_disp={command['red_object_displacement']}"
        )

    print(f"[SUMMARY] best_command={report['best_command']['name']}")
    print(f"[SUMMARY] any_red_grasper_contact={report['flags']['any_red_grasper_contact']}")
    print(f"[SUMMARY] judgment={report['judgment']}")

    try:
        saved_path = write_scan_report(report, output_path)
    except Exception as exc:
        print(f"[FAIL] failed to write diagnostics JSON: {exc}")
        return 4

    print(f"[OK] wrote {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
