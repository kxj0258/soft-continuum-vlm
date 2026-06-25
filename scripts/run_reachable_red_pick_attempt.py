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
FULL_SECTION_ANGLES = [2.1, 1.5708, 1.5, 1.5708, 1.0, 1.5708]
PRECLOSE_SCALE = 0.7
CLOSE_ADVANCE_SCALE = 0.0
GRASPER_ROTATION = -0.7854


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reachable red-object pick-attempt contact diagnostic."
    )
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--approach-ramp-steps", type=int, default=120)
    parser.add_argument("--hold-preclose-steps", type=int, default=160)
    parser.add_argument("--close-ramp-steps", type=int, default=120)
    parser.add_argument("--hold-closed-steps", type=int, default=240)
    parser.add_argument("--relax-hold-steps", type=int, default=160)
    parser.add_argument("--include-relax", action="store_true", default=False)
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


def build_preclose_section_angles(
    *,
    full_section_angles: Sequence[float] = FULL_SECTION_ANGLES,
    preclose_scale: float = PRECLOSE_SCALE,
) -> list[float]:
    values = _as_float_sequence(full_section_angles, "full_section_angles")
    if len(values) != 6:
        raise ValueError("full_section_angles must contain 6 values.")
    return [
        float(values[0] * preclose_scale),
        float(values[1]),
        float(values[2] * preclose_scale),
        float(values[3]),
        float(values[4] * preclose_scale),
        float(values[5]),
    ]


def build_phase_schedule(*, include_relax: bool) -> list[dict[str, Any]]:
    schedule = [
        {"name": "approach_ramp", "steps": 120},
        {"name": "hold_preclose", "steps": 160},
        {"name": "close_ramp", "steps": 120},
        {"name": "hold_closed", "steps": 240},
    ]
    if include_relax:
        schedule.append({"name": "relax_hold", "steps": 160})
    return schedule


def compute_contact_ratio(contact_steps: Sequence[bool]) -> float:
    steps = list(contact_steps)
    if not steps:
        return 0.0
    return float(sum(1 for step in steps if step) / len(steps))


def compute_success_flags(
    *,
    hold_preclose_has_contact: bool,
    close_has_contact: bool,
    hold_closed_has_contact: bool,
    hold_closed_contact_ratio: float,
    red_displacement_after_close: float,
) -> dict[str, bool]:
    clean_preclose = not bool(hold_preclose_has_contact)
    close_added_contact = bool(close_has_contact or hold_closed_has_contact)
    hold_closed_contact_persistent = bool(float(hold_closed_contact_ratio) >= 0.25)
    red_object_moved_after_close = bool(float(red_displacement_after_close) > 0.005)
    return {
        "clean_preclose": bool(clean_preclose),
        "close_added_contact": bool(close_added_contact),
        "hold_closed_contact_persistent": bool(hold_closed_contact_persistent),
        "red_object_moved_after_close": bool(red_object_moved_after_close),
    }


def make_judgment(success_flags: dict[str, bool]) -> str:
    if (
        bool(success_flags["clean_preclose"])
        and bool(success_flags["close_added_contact"])
        and bool(success_flags["hold_closed_contact_persistent"])
    ):
        return "[OK] reproducible clean-close red contact attempt"
    if bool(success_flags["clean_preclose"]) and bool(success_flags["close_added_contact"]):
        return "[WARN] close contact occurred but was not persistent"
    if not bool(success_flags["clean_preclose"]):
        return "[WARN] preclose already had contact; not a clean pick attempt"
    return "[FAIL] close did not produce red contact"


def summarize_phase(phase_name: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    phase_records = [record for record in records if record["phase"] == phase_name]
    if not phase_records:
        return {
            "phase": phase_name,
            "steps": 0,
            "min_tip_red_distance": None,
            "final_tip_red_distance": None,
            "has_red_grasper_contact": False,
            "contact_step_count": 0,
            "contact_ratio": 0.0,
            "max_red_grasper_normal_force": 0.0,
            "mean_red_grasper_normal_force": 0.0,
            "red_displacement_start": 0.0,
            "red_displacement_end": 0.0,
            "red_displacement_delta": 0.0,
        }

    contact_records = [record for record in phase_records if bool(record["has_red_grasper_contact"])]
    contact_forces = [float(record["red_grasper_normal_force"]) for record in contact_records]
    red_displacement_start = float(phase_records[0]["red_displacement"])
    red_displacement_end = float(phase_records[-1]["red_displacement"])

    return {
        "phase": phase_name,
        "steps": int(len(phase_records)),
        "min_tip_red_distance": float(min(record["tip_red_distance"] for record in phase_records)),
        "final_tip_red_distance": float(phase_records[-1]["tip_red_distance"]),
        "has_red_grasper_contact": bool(len(contact_records) > 0),
        "contact_step_count": int(len(contact_records)),
        "contact_ratio": compute_contact_ratio([bool(record["has_red_grasper_contact"]) for record in phase_records]),
        "max_red_grasper_normal_force": float(max(contact_forces)) if contact_forces else 0.0,
        "mean_red_grasper_normal_force": float(sum(contact_forces) / len(contact_forces)) if contact_forces else 0.0,
        "red_displacement_start": red_displacement_start,
        "red_displacement_end": red_displacement_end,
        "red_displacement_delta": float(red_displacement_end - red_displacement_start),
    }


def _distance(position_a: Sequence[float], position_b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(position_a, dtype=np.float64) - np.asarray(position_b, dtype=np.float64)))


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
        raise RuntimeError("Missing Feagine wrapper section angle interface: drive_section_angles/section_count.")
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
                "distance": float(contact.dist),
                "normal_force": 0.0 if force6 is None else float(force6[0]),
            }
        )
    return contacts


def _classify_contacts(contacts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    has_red_grasper_contact = False
    has_red_pedestal_contact = False
    has_red_table_contact = False
    red_grasper_forces: list[float] = []
    for contact in contacts:
        geom1 = str(contact["geom1"])
        geom2 = str(contact["geom2"])
        body1 = str(contact["body1"])
        body2 = str(contact["body2"])
        labels = [geom1, geom2, body1, body2]
        has_red = any("red_object" in label.lower() for label in labels)
        has_pedestal = any("red_pedestal" in label.lower() for label in labels)
        has_table = any("tabletop" in label.lower() or "table" in label.lower() for label in labels)
        has_grasper = any(
            needle in body.lower()
            for body in (body1, body2)
            for needle in ("feagine_grasper", "grasper", "finger")
        )
        has_red_pedestal_contact = has_red_pedestal_contact or (has_red and has_pedestal)
        has_red_table_contact = has_red_table_contact or (has_red and has_table)
        if has_red and has_grasper:
            has_red_grasper_contact = True
            red_grasper_forces.append(abs(float(contact["normal_force"])))
    return {
        "has_red_grasper_contact": bool(has_red_grasper_contact),
        "has_red_pedestal_contact": bool(has_red_pedestal_contact),
        "has_red_table_contact": bool(has_red_table_contact),
        "red_grasper_normal_force": float(max(red_grasper_forces)) if red_grasper_forces else 0.0,
    }


def _interpolate_section_angles(target_section_angles: Sequence[float], *, step_index: int, total_steps: int) -> list[float]:
    if total_steps <= 0:
        return [float(value) for value in target_section_angles]
    alpha = float(step_index + 1) / float(total_steps)
    return [float(alpha * float(value)) for value in target_section_angles]


def _interpolate_rotation(start_rotation: float, end_rotation: float, *, step_index: int, total_steps: int) -> float:
    if total_steps <= 0:
        return float(end_rotation)
    alpha = float(step_index + 1) / float(total_steps)
    return float((1.0 - alpha) * float(start_rotation) + alpha * float(end_rotation))


def _interpolate_grip_command(*, step_index: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    return float((step_index + 1) / float(total_steps))


def _record_step(
    *,
    phase: str,
    step_index: int,
    mujoco: Any,
    model: Any,
    data: Any,
    initial_red_position: Sequence[float],
    grip_command: float,
    section_angles: Sequence[float],
    grasper_rotation: float,
) -> dict[str, Any]:
    tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    red_position = _body_position(mujoco, model, data, "red_object")
    if tip_position is None or red_position is None:
        raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")
    contacts = _contact_snapshot(mujoco, model, data)
    flags = _classify_contacts(contacts)
    return {
        "phase": phase,
        "step": int(step_index),
        "tip_position": [float(value) for value in tip_position],
        "red_position": [float(value) for value in red_position],
        "red_displacement": float(_distance(initial_red_position, red_position)),
        "tip_red_distance": float(_distance(tip_position, red_position)),
        "grip_command": float(grip_command),
        "section_angles": [float(value) for value in section_angles],
        "grasper_rotation": float(grasper_rotation),
        "contact_pairs": sorted(
            {" <-> ".join(sorted([str(contact["geom1"]), str(contact["geom2"])])) for contact in contacts}
        ),
        "has_red_grasper_contact": bool(flags["has_red_grasper_contact"]),
        "has_red_pedestal_contact": bool(flags["has_red_pedestal_contact"]),
        "has_red_table_contact": bool(flags["has_red_table_contact"]),
        "red_grasper_normal_force": float(flags["red_grasper_normal_force"]),
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
                step_index=step_index,
                mujoco=mujoco,
                model=model,
                data=data,
                initial_red_position=initial_red_position,
                grip_command=float(action["grip_command"]),
                section_angles=action["section_angles"],
                grasper_rotation=float(action["grasper_rotation"]),
            )
        )


def run_pick_attempt(
    scene_path: str | Path,
    *,
    settle_steps: int,
    approach_ramp_steps: int,
    hold_preclose_steps: int,
    close_ramp_steps: int,
    hold_closed_steps: int,
    relax_hold_steps: int,
    include_relax: bool,
    save_trajectory: bool,
) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model, data = _load_tabletop_scene(str(resolved_scene))

    try:
        robot = _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc

    preclose_section_angles = build_preclose_section_angles(
        full_section_angles=FULL_SECTION_ANGLES,
        preclose_scale=PRECLOSE_SCALE,
    )
    schedule = build_phase_schedule(include_relax=include_relax)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(int(max(0, settle_steps))):
        mujoco.mj_step(model, data)

    initial_red_position = _body_position(mujoco, model, data, "red_object")
    initial_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    if initial_red_position is None or initial_tip_position is None:
        raise RuntimeError("Missing red_object or feagine_grasper_tip body pose.")

    records: list[dict[str, Any]] = []
    _run_phase(
        phase_name="approach_ramp",
        steps=approach_ramp_steps,
        robot=robot,
        mujoco=mujoco,
        model=model,
        data=data,
        initial_red_position=initial_red_position,
        records=records,
        action_factory=lambda step_index, total_steps: {
            "section_angles": _interpolate_section_angles(
                preclose_section_angles,
                step_index=step_index,
                total_steps=total_steps,
            ),
            "grip_command": 0.0,
            "grasper_rotation": _interpolate_rotation(
                0.0,
                GRASPER_ROTATION,
                step_index=step_index,
                total_steps=total_steps,
            ),
        },
    )
    _run_phase(
        phase_name="hold_preclose",
        steps=hold_preclose_steps,
        robot=robot,
        mujoco=mujoco,
        model=model,
        data=data,
        initial_red_position=initial_red_position,
        records=records,
        action_factory=lambda step_index, total_steps: {
            "section_angles": [float(value) for value in preclose_section_angles],
            "grip_command": 0.0,
            "grasper_rotation": float(GRASPER_ROTATION),
        },
    )
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
            "section_angles": [float(value) for value in preclose_section_angles],
            "grip_command": _interpolate_grip_command(step_index=step_index, total_steps=total_steps),
            "grasper_rotation": float(GRASPER_ROTATION),
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
            "section_angles": [float(value) for value in preclose_section_angles],
            "grip_command": 1.0,
            "grasper_rotation": float(GRASPER_ROTATION),
        },
    )
    if include_relax:
        _run_phase(
            phase_name="relax_hold",
            steps=relax_hold_steps,
            robot=robot,
            mujoco=mujoco,
            model=model,
            data=data,
            initial_red_position=initial_red_position,
            records=records,
            action_factory=lambda step_index, total_steps: {
                "section_angles": [float(value) for value in preclose_section_angles],
                "grip_command": 1.0,
                "grasper_rotation": float(GRASPER_ROTATION),
            },
        )

    final_red_position = _body_position(mujoco, model, data, "red_object")
    final_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    if final_red_position is None or final_tip_position is None:
        raise RuntimeError("Missing red_object or feagine_grasper_tip body pose.")

    phase_summaries = {phase["name"]: summarize_phase(phase["name"], records) for phase in schedule}
    hold_closed_summary = phase_summaries["hold_closed"]
    close_summary = phase_summaries["close_ramp"]
    hold_preclose_summary = phase_summaries["hold_preclose"]

    all_contact_pairs = sorted({pair for record in records for pair in record["contact_pairs"]})
    first_contact_record = next((record for record in records if bool(record["has_red_grasper_contact"])), None)
    overall_forces = [
        float(record["red_grasper_normal_force"])
        for record in records
        if bool(record["has_red_grasper_contact"])
    ]
    total_red_displacement = float(_distance(initial_red_position, final_red_position))
    red_displacement_after_close = max(
        float(close_summary["red_displacement_end"]),
        float(hold_closed_summary["red_displacement_end"]),
    ) - float(hold_preclose_summary["red_displacement_end"])
    success_flags = compute_success_flags(
        hold_preclose_has_contact=bool(hold_preclose_summary["has_red_grasper_contact"]),
        close_has_contact=bool(close_summary["has_red_grasper_contact"]),
        hold_closed_has_contact=bool(hold_closed_summary["has_red_grasper_contact"]),
        hold_closed_contact_ratio=float(hold_closed_summary["contact_ratio"]),
        red_displacement_after_close=float(red_displacement_after_close),
    )
    judgment = make_judgment(success_flags)

    report = {
        "scene": str(resolved_scene),
        "target": {
            "full_section_angles": [float(value) for value in FULL_SECTION_ANGLES],
            "preclose_scale": float(PRECLOSE_SCALE),
            "preclose_section_angles": [float(value) for value in preclose_section_angles],
            "close_advance_scale": float(CLOSE_ADVANCE_SCALE),
            "grasper_rotation": float(GRASPER_ROTATION),
        },
        "initial_red_position": [float(value) for value in initial_red_position],
        "final_red_position": [float(value) for value in final_red_position],
        "initial_tip_position": [float(value) for value in initial_tip_position],
        "final_tip_position": [float(value) for value in final_tip_position],
        "phase_summaries": phase_summaries,
        "overall": {
            "min_tip_red_distance": float(min(record["tip_red_distance"] for record in records)),
            "max_red_displacement": float(max(record["red_displacement"] for record in records)),
            "total_red_displacement": float(total_red_displacement),
            "any_red_grasper_contact": any(bool(record["has_red_grasper_contact"]) for record in records),
            "contact_first_phase": None if first_contact_record is None else str(first_contact_record["phase"]),
            "contact_first_step": None if first_contact_record is None else int(first_contact_record["step"]),
            "hold_closed_contact_ratio": float(hold_closed_summary["contact_ratio"]),
            "max_red_grasper_normal_force": float(max(overall_forces)) if overall_forces else 0.0,
            "mean_red_grasper_normal_force": float(sum(overall_forces) / len(overall_forces)) if overall_forces else 0.0,
            "contact_pairs_seen": all_contact_pairs,
        },
        "success_flags": success_flags,
        "judgment": judgment,
        "notes": [
            "This is a contact-based pick attempt diagnostic.",
            "It does not claim stable grasp or object lifting.",
        ],
    }
    if save_trajectory:
        report["trajectory"] = records
    return report


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        report = run_pick_attempt(
            scene_path,
            settle_steps=args.settle_steps,
            approach_ramp_steps=args.approach_ramp_steps,
            hold_preclose_steps=args.hold_preclose_steps,
            close_ramp_steps=args.close_ramp_steps,
            hold_closed_steps=args.hold_closed_steps,
            relax_hold_steps=args.relax_hold_steps,
            include_relax=bool(args.include_relax),
            save_trajectory=bool(args.save_trajectory),
        )
    except ValueError as exc:
        if "unsupported action keys" in str(exc):
            print(f"[FAIL] {exc}")
            return 4
        print(f"[FAIL] pick attempt failed: {exc}")
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
        print(f"[FAIL] pick attempt failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] pick attempt failed: {exc}")
        return 1

    for phase_name in report["phase_summaries"]:
        summary = report["phase_summaries"][phase_name]
        print(
            f"[PHASE] {phase_name} "
            f"steps={summary['steps']} "
            f"contact_ratio={summary['contact_ratio']} "
            f"max_force={summary['max_red_grasper_normal_force']}"
        )

    print(f"[RESULT] clean_preclose={report['success_flags']['clean_preclose']}")
    print(f"[RESULT] close_added_contact={report['success_flags']['close_added_contact']}")
    print(f"[RESULT] hold_closed_contact_ratio={report['overall']['hold_closed_contact_ratio']}")
    print(f"[RESULT] red_object_moved_after_close={report['success_flags']['red_object_moved_after_close']}")
    print(f"[RESULT] max_red_grasper_normal_force={report['overall']['max_red_grasper_normal_force']}")
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
