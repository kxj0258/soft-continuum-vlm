from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


ALLOWED_ACTION_KEYS = {"section_angles", "grip_command", "grasper_rotation"}
TARGET_SECTION_ANGLES = [2.1, 1.5708, 1.5, 1.5708, 1.0, 1.5708]
TARGET_GRASPER_ROTATION = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose controlled close contact after a best-local open approach."
    )
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--approach-steps", type=int, default=160)
    parser.add_argument("--hold-open-steps", type=int, default=80)
    parser.add_argument("--close-ramp-steps", type=int, default=80)
    parser.add_argument("--hold-closed-steps", type=int, default=120)
    parser.add_argument("--hold-open-baseline-steps", type=int, default=200)
    parser.add_argument("--save-trajectory", action="store_true", default=False)
    return parser.parse_args()


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
    for index in (0, 2, 4):
        value = float(section_angles[index])
        if value < 0.0 or value > 2.356:
            raise ValueError("bend magnitude must be in [0, 2.356].")
    for index in (1, 3, 5):
        value = float(section_angles[index])
        if value < -math.pi or value > math.pi:
            raise ValueError("direction angle must be in [-pi, pi].")
    float(action["grip_command"])
    float(action["grasper_rotation"])


def interpolate_section_angles(
    target_section_angles: Sequence[float],
    *,
    step_index: int,
    total_steps: int,
) -> list[float]:
    if total_steps <= 0:
        return [float(value) for value in target_section_angles]
    alpha = float(step_index + 1) / float(total_steps)
    return [float(alpha * float(value)) for value in target_section_angles]


def interpolate_grip_command(*, step_index: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    return float((step_index + 1) / float(total_steps))


def _distance(position_a: Sequence[float], position_b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(position_a, dtype=np.float64) - np.asarray(position_b, dtype=np.float64)))


def _name_matches(value: str, needles: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(needle in lowered for needle in needles)


def classify_contacts(contacts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    has_red_grasper_contact = False
    has_red_pedestal_contact = False
    has_red_table_contact = False
    max_red_grasper_normal_force = 0.0
    total_red_grasper_normal_force = 0.0

    for contact in contacts:
        geom1 = str(contact.get("geom1", ""))
        geom2 = str(contact.get("geom2", ""))
        body1 = str(contact.get("body1", ""))
        body2 = str(contact.get("body2", ""))
        normal_force = abs(float(contact.get("normal_force", 0.0) or 0.0))

        labels = [geom1, geom2, body1, body2]
        has_red = any(_name_matches(label, ("red_object",)) for label in labels)
        has_pedestal = any(_name_matches(label, ("red_pedestal",)) for label in labels)
        has_table = any(_name_matches(label, ("tabletop", "table")) for label in labels)
        has_grasper = any(_name_matches(label, ("grasper", "finger", "feagine_grasper")) for label in labels)

        has_red_pedestal_contact = has_red_pedestal_contact or (has_red and has_pedestal)
        has_red_table_contact = has_red_table_contact or (has_red and has_table)
        if has_red and has_grasper:
            has_red_grasper_contact = True
            max_red_grasper_normal_force = max(max_red_grasper_normal_force, normal_force)
            total_red_grasper_normal_force += normal_force

    return {
        "has_red_grasper_contact": bool(has_red_grasper_contact),
        "has_red_pedestal_contact": bool(has_red_pedestal_contact),
        "has_red_table_contact": bool(has_red_table_contact),
        "max_red_grasper_normal_force": float(max_red_grasper_normal_force),
        "total_red_grasper_normal_force": float(total_red_grasper_normal_force),
    }


def summarize_trial(
    *,
    name: str,
    initial_tip_position: Sequence[float],
    final_tip_position: Sequence[float],
    initial_red_position: Sequence[float],
    final_red_position: Sequence[float],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    min_tip_red_distance = min(float(record["tip_red_distance"]) for record in records)
    final_tip_red_distance = float(records[-1]["tip_red_distance"]) if records else float("inf")
    max_red_displacement = max(float(record["red_displacement"]) for record in records) if records else 0.0
    max_red_grasper_normal_force = max(
        float(record["max_red_grasper_normal_force"]) for record in records
    ) if records else 0.0
    contact_pairs_seen = sorted(
        {
            pair
            for record in records
            for pair in record.get("contact_pairs", [])
        }
    )

    def phase_max_displacement(phase_name: str) -> float:
        values = [float(record["red_displacement"]) for record in records if record["phase"] == phase_name]
        return float(max(values)) if values else 0.0

    def phase_contact(phase_name: str) -> bool:
        return any(bool(record["has_red_grasper_contact"]) for record in records if record["phase"] == phase_name)

    return {
        "name": name,
        "initial_red_position": [float(value) for value in initial_red_position],
        "final_red_position": [float(value) for value in final_red_position],
        "initial_tip_position": [float(value) for value in initial_tip_position],
        "final_tip_position": [float(value) for value in final_tip_position],
        "min_tip_red_distance": float(min_tip_red_distance),
        "final_tip_red_distance": float(final_tip_red_distance),
        "max_red_displacement": float(max_red_displacement),
        "red_displacement_during_approach": phase_max_displacement("approach_ramp"),
        "red_displacement_during_close": phase_max_displacement("close_ramp"),
        "red_displacement_during_hold_closed": phase_max_displacement("hold_closed"),
        "has_red_grasper_contact_during_approach": phase_contact("approach_ramp"),
        "has_red_grasper_contact_during_close": phase_contact("close_ramp"),
        "has_red_grasper_contact_during_hold_closed": phase_contact("hold_closed"),
        "max_red_grasper_normal_force": float(max_red_grasper_normal_force),
        "contact_pairs_seen": contact_pairs_seen,
    }


def compare_trials(*, open_only_baseline: dict[str, Any], controlled_close: dict[str, Any]) -> dict[str, Any]:
    close_added_contact = bool(
        (
            bool(controlled_close.get("has_red_grasper_contact_during_close"))
            or bool(controlled_close.get("has_red_grasper_contact_during_hold_closed"))
        )
        and not (
            bool(open_only_baseline.get("has_red_grasper_contact_during_close"))
            or bool(open_only_baseline.get("has_red_grasper_contact_during_hold_closed"))
        )
    )
    return {
        "close_added_red_displacement": float(controlled_close["max_red_displacement"]) - float(open_only_baseline["max_red_displacement"]),
        "close_added_contact": close_added_contact,
        "close_max_force_minus_baseline": float(controlled_close["max_red_grasper_normal_force"]) - float(open_only_baseline["max_red_grasper_normal_force"]),
        "close_min_distance_minus_baseline": float(controlled_close["min_tip_red_distance"]) - float(open_only_baseline["min_tip_red_distance"]),
    }


def make_judgment(
    *,
    open_only_baseline: dict[str, Any],
    controlled_close: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    if (
        bool(open_only_baseline.get("has_red_grasper_contact_during_approach"))
        and bool(controlled_close.get("has_red_grasper_contact_during_approach"))
        and (bool(controlled_close.get("has_red_grasper_contact_during_close")) or bool(controlled_close.get("has_red_grasper_contact_during_hold_closed")))
        and not bool(comparison["close_added_contact"])
        and float(comparison["close_added_red_displacement"]) <= 0.005
        and float(comparison["close_max_force_minus_baseline"]) <= 0.05
    ):
        return "[WARN] approach produced contact before close; close effect is ambiguous"
    if (
        (bool(controlled_close.get("has_red_grasper_contact_during_close")) or bool(controlled_close.get("has_red_grasper_contact_during_hold_closed")))
        and (
            float(comparison["close_added_red_displacement"]) > 1e-4
            or float(comparison["close_max_force_minus_baseline"]) > 1e-4
            or bool(comparison["close_added_contact"])
        )
    ):
        return "[OK] controlled close produced red-grasper contact"
    if bool(open_only_baseline.get("has_red_grasper_contact_during_approach")) or bool(controlled_close.get("has_red_grasper_contact_during_approach")):
        return "[WARN] close did not add contact beyond approach"
    return "[FAIL] no red-grasper contact observed"


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


def _contact_force6(mujoco: Any, model: Any, data: Any, index: int) -> list[float] | None:
    force = np.zeros(6, dtype=np.float64)
    try:
        mujoco.mj_contactForce(model, data, index, force)
    except Exception:
        return None
    return [float(item) for item in force.tolist()]


def _contact_snapshot(mujoco: Any, model: Any, data: Any) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        body1_id = int(model.geom_bodyid[geom1_id])
        body2_id = int(model.geom_bodyid[geom2_id])
        force6 = _contact_force6(mujoco, model, data, index)
        contacts.append(
            {
                "geom1": _geom_name(model, geom1_id),
                "geom2": _geom_name(model, geom2_id),
                "body1": _body_name(model, body1_id),
                "body2": _body_name(model, body2_id),
                "normal_force": None if force6 is None else float(force6[0]),
            }
        )
    return contacts


def _record_step(
    *,
    phase: str,
    mujoco: Any,
    model: Any,
    data: Any,
    initial_red_position: Sequence[float],
    grip_command: float,
    section_angles: Sequence[float],
) -> dict[str, Any]:
    tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    red_position = _body_position(mujoco, model, data, "red_object")
    if tip_position is None or red_position is None:
        raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")
    contacts = _contact_snapshot(mujoco, model, data)
    flags = classify_contacts(contacts)
    return {
        "phase": phase,
        "tip_position": [float(value) for value in tip_position],
        "red_position": [float(value) for value in red_position],
        "tip_red_distance": float(_distance(tip_position, red_position)),
        "red_displacement": float(_distance(initial_red_position, red_position)),
        "grip_command": float(grip_command),
        "section_angles": [float(value) for value in section_angles],
        "contact_pairs": sorted(
            {
                " <-> ".join(sorted([str(contact["geom1"]), str(contact["geom2"])]))
                for contact in contacts
            }
        ),
        "has_red_grasper_contact": bool(flags["has_red_grasper_contact"]),
        "has_red_pedestal_contact": bool(flags["has_red_pedestal_contact"]),
        "has_red_table_contact": bool(flags["has_red_table_contact"]),
        "max_red_grasper_normal_force": float(flags["max_red_grasper_normal_force"]),
        "total_red_grasper_normal_force": float(flags["total_red_grasper_normal_force"]),
        "num_contacts": int(data.ncon),
    }


def _run_phase(
    *,
    phase_name: str,
    steps: int,
    robot: Any,
    mujoco: Any,
    model: Any,
    data: Any,
    initial_red_position: Sequence[float],
    action_factory,
    records: list[dict[str, Any]],
) -> None:
    for step_index in range(int(max(0, steps))):
        action = action_factory(step_index, steps)
        _apply_feagine_action(robot, action)
        mujoco.mj_step(model, data)
        records.append(
            _record_step(
                phase=phase_name,
                mujoco=mujoco,
                model=model,
                data=data,
                initial_red_position=initial_red_position,
                grip_command=float(action["grip_command"]),
                section_angles=action["section_angles"],
            )
        )


def run_trial(
    *,
    name: str,
    model: Any,
    data: Any,
    mujoco: Any,
    settle_steps: int,
    approach_steps: int,
    hold_open_steps: int,
    close_ramp_steps: int,
    hold_closed_steps: int,
    hold_open_baseline_steps: int,
    save_trajectory: bool,
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
    initial_red_position = _body_position(mujoco, model, data, "red_object")
    if initial_tip_position is None or initial_red_position is None:
        raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")

    records: list[dict[str, Any]] = []

    _run_phase(
        phase_name="approach_ramp",
        steps=approach_steps,
        robot=robot,
        mujoco=mujoco,
        model=model,
        data=data,
        initial_red_position=initial_red_position,
        records=records,
        action_factory=lambda step_index, total_steps: {
            "section_angles": interpolate_section_angles(TARGET_SECTION_ANGLES, step_index=step_index, total_steps=total_steps),
            "grip_command": 0.0,
            "grasper_rotation": TARGET_GRASPER_ROTATION,
        },
    )

    _run_phase(
        phase_name="hold_open",
        steps=hold_open_steps,
        robot=robot,
        mujoco=mujoco,
        model=model,
        data=data,
        initial_red_position=initial_red_position,
        records=records,
        action_factory=lambda step_index, total_steps: {
            "section_angles": [float(value) for value in TARGET_SECTION_ANGLES],
            "grip_command": 0.0,
            "grasper_rotation": TARGET_GRASPER_ROTATION,
        },
    )

    if name == "controlled_close":
        _run_phase(
            phase_name="close_ramp",
            steps=close_ramp_steps,
            robot=robot,
            mujoco=mujoco,
            model=model,
            data=data,
            initial_red_position=initial_red_position,
            records=records,
            action_factory=lambda step_index, total_steps: {
                "section_angles": [float(value) for value in TARGET_SECTION_ANGLES],
                "grip_command": interpolate_grip_command(step_index=step_index, total_steps=total_steps),
                "grasper_rotation": TARGET_GRASPER_ROTATION,
            },
        )
        _run_phase(
            phase_name="hold_closed",
            steps=hold_closed_steps,
            robot=robot,
            mujoco=mujoco,
            model=model,
            data=data,
            initial_red_position=initial_red_position,
            records=records,
            action_factory=lambda step_index, total_steps: {
                "section_angles": [float(value) for value in TARGET_SECTION_ANGLES],
                "grip_command": 1.0,
                "grasper_rotation": TARGET_GRASPER_ROTATION,
            },
        )
    else:
        _run_phase(
            phase_name="hold_open_baseline",
            steps=hold_open_baseline_steps,
            robot=robot,
            mujoco=mujoco,
            model=model,
            data=data,
            initial_red_position=initial_red_position,
            records=records,
            action_factory=lambda step_index, total_steps: {
                "section_angles": [float(value) for value in TARGET_SECTION_ANGLES],
                "grip_command": 0.0,
                "grasper_rotation": TARGET_GRASPER_ROTATION,
            },
        )

    final_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    final_red_position = _body_position(mujoco, model, data, "red_object")
    if final_tip_position is None or final_red_position is None:
        raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")

    summary = summarize_trial(
        name=name,
        initial_tip_position=initial_tip_position,
        final_tip_position=final_tip_position,
        initial_red_position=initial_red_position,
        final_red_position=final_red_position,
        records=records,
    )
    if save_trajectory:
        summary["trajectory"] = records
    return summary


def run_controlled_close_diagnostic(
    scene_path: str | Path,
    *,
    settle_steps: int,
    approach_steps: int,
    hold_open_steps: int,
    close_ramp_steps: int,
    hold_closed_steps: int,
    hold_open_baseline_steps: int,
    save_trajectory: bool,
) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model, data = _load_tabletop_scene(str(resolved_scene))

    try:
        _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc

    open_only_baseline = run_trial(
        name="open_only_baseline",
        model=model,
        data=data,
        mujoco=mujoco,
        settle_steps=settle_steps,
        approach_steps=approach_steps,
        hold_open_steps=hold_open_steps,
        close_ramp_steps=close_ramp_steps,
        hold_closed_steps=hold_closed_steps,
        hold_open_baseline_steps=hold_open_baseline_steps,
        save_trajectory=save_trajectory,
    )
    controlled_close = run_trial(
        name="controlled_close",
        model=model,
        data=data,
        mujoco=mujoco,
        settle_steps=settle_steps,
        approach_steps=approach_steps,
        hold_open_steps=hold_open_steps,
        close_ramp_steps=close_ramp_steps,
        hold_closed_steps=hold_closed_steps,
        hold_open_baseline_steps=hold_open_baseline_steps,
        save_trajectory=save_trajectory,
    )
    comparison = compare_trials(
        open_only_baseline=open_only_baseline,
        controlled_close=controlled_close,
    )
    judgment = make_judgment(
        open_only_baseline=open_only_baseline,
        controlled_close=controlled_close,
        comparison=comparison,
    )

    return {
        "scene": str(resolved_scene),
        "target_action": {
            "section_angles": [float(value) for value in TARGET_SECTION_ANGLES],
            "grasper_rotation": float(TARGET_GRASPER_ROTATION),
        },
        "trials": [open_only_baseline, controlled_close],
        "comparison": comparison,
        "judgment": judgment,
        "notes": [
            "This script diagnoses controlled gripper close contact.",
            "It does not claim grasp success or object lifting.",
        ],
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def _trial_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    for trial in report["trials"]:
        if trial["name"] == name:
            return trial
    raise KeyError(name)


def main() -> int:
    args = _parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        report = run_controlled_close_diagnostic(
            scene_path,
            settle_steps=args.settle_steps,
            approach_steps=args.approach_steps,
            hold_open_steps=args.hold_open_steps,
            close_ramp_steps=args.close_ramp_steps,
            hold_closed_steps=args.hold_closed_steps,
            hold_open_baseline_steps=args.hold_open_baseline_steps,
            save_trajectory=bool(args.save_trajectory),
        )
    except ValueError as exc:
        if "unsupported action keys" in str(exc):
            print(f"[FAIL] {exc}")
            return 4
        print(f"[FAIL] controlled close test failed: {exc}")
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if "wrapper creation failed" in message:
            print("[FAIL] Failed to create Feagine wrapper for tabletop scene.")
            print(f"[DETAIL] {message.removeprefix('wrapper creation failed: ')}")
            return 2
        if "Missing Feagine wrapper" in message or "section_angles" in message:
            print("[FAIL] Could not apply Feagine action to tabletop scene.")
            print(f"[DETAIL] {message}")
            return 3
        print(f"[FAIL] controlled close test failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] controlled close test failed: {exc}")
        return 1

    open_only = _trial_by_name(report, "open_only_baseline")
    controlled_close = _trial_by_name(report, "controlled_close")
    comparison = report["comparison"]

    print("[TRIAL] open_only_baseline")
    print(
        "[RESULT] open_only "
        f"min_dist={open_only['min_tip_red_distance']} "
        f"max_red_disp={open_only['max_red_displacement']} "
        f"contact_approach={open_only['has_red_grasper_contact_during_approach']} "
        "contact_close=False"
    )
    print()
    print("[TRIAL] controlled_close")
    print(
        "[RESULT] controlled_close "
        f"min_dist={controlled_close['min_tip_red_distance']} "
        f"max_red_disp={controlled_close['max_red_displacement']} "
        f"contact_approach={controlled_close['has_red_grasper_contact_during_approach']} "
        f"contact_close={controlled_close['has_red_grasper_contact_during_close']} "
        f"contact_hold_closed={controlled_close['has_red_grasper_contact_during_hold_closed']}"
    )
    print()
    print(f"[COMPARISON] close_added_red_displacement={comparison['close_added_red_displacement']}")
    print(f"[COMPARISON] close_added_contact={comparison['close_added_contact']}")
    print(f"[RESULT] judgment={report['judgment']}")

    try:
        saved_path = write_report(report, output_path)
    except Exception as exc:
        print(f"[FAIL] failed to write diagnostics JSON: {exc}")
        return 1

    print(f"[OK] wrote {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
