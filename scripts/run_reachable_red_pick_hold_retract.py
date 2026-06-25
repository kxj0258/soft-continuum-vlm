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
GRASPER_ROTATION = -0.7854


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reachable red-object hold/retract contact diagnostics."
    )
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--output", required=True, help="Path to the diagnostics JSON.")
    parser.add_argument("--settle-steps", type=int, default=80)
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


def scale_section_angles(section_angles: Sequence[float], *, scale: float) -> list[float]:
    values = _as_float_sequence(section_angles, "section_angles")
    if len(values) != 6:
        raise ValueError("section_angles must contain 6 values.")
    return [
        float(values[0] * scale),
        float(values[1]),
        float(values[2] * scale),
        float(values[3]),
        float(values[4] * scale),
        float(values[5]),
    ]


def build_trial_schedules() -> list[dict[str, Any]]:
    shared_phases = [
        {
            "name": "approach_ramp",
            "steps": 120,
            "scale_start": 0.0,
            "scale_end": PRECLOSE_SCALE,
            "grip_start": 0.0,
            "grip_end": 0.0,
        },
        {
            "name": "hold_preclose",
            "steps": 160,
            "scale_start": PRECLOSE_SCALE,
            "scale_end": PRECLOSE_SCALE,
            "grip_start": 0.0,
            "grip_end": 0.0,
        },
        {
            "name": "close_ramp",
            "steps": 120,
            "scale_start": PRECLOSE_SCALE,
            "scale_end": PRECLOSE_SCALE,
            "grip_start": 0.0,
            "grip_end": 1.0,
        },
        {
            "name": "hold_closed_initial",
            "steps": 240,
            "scale_start": PRECLOSE_SCALE,
            "scale_end": PRECLOSE_SCALE,
            "grip_start": 1.0,
            "grip_end": 1.0,
        },
    ]
    return [
        {
            "name": "extended_hold",
            "phases": shared_phases
            + [
                {
                    "name": "extended_hold",
                    "steps": 600,
                    "scale_start": PRECLOSE_SCALE,
                    "scale_end": PRECLOSE_SCALE,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                }
            ],
        },
        {
            "name": "gentle_retract_downscale",
            "phases": shared_phases
            + [
                {
                    "name": "retract_ramp",
                    "steps": 160,
                    "scale_start": 0.70,
                    "scale_end": 0.60,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                },
                {
                    "name": "retract_hold",
                    "steps": 300,
                    "scale_start": 0.60,
                    "scale_end": 0.60,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                },
            ],
        },
        {
            "name": "gentle_advance_upscale",
            "phases": shared_phases
            + [
                {
                    "name": "advance_ramp",
                    "steps": 160,
                    "scale_start": 0.70,
                    "scale_end": 0.75,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                },
                {
                    "name": "advance_hold",
                    "steps": 300,
                    "scale_start": 0.75,
                    "scale_end": 0.75,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                },
            ],
        },
        {
            "name": "regrip_squeeze",
            "phases": shared_phases
            + [
                {
                    "name": "partial_open",
                    "steps": 80,
                    "scale_start": PRECLOSE_SCALE,
                    "scale_end": PRECLOSE_SCALE,
                    "grip_start": 1.0,
                    "grip_end": 0.6,
                },
                {
                    "name": "reclose",
                    "steps": 120,
                    "scale_start": PRECLOSE_SCALE,
                    "scale_end": PRECLOSE_SCALE,
                    "grip_start": 0.6,
                    "grip_end": 1.0,
                },
                {
                    "name": "regrip_hold",
                    "steps": 300,
                    "scale_start": PRECLOSE_SCALE,
                    "scale_end": PRECLOSE_SCALE,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                },
            ],
        },
    ]


def compute_post_close_contact_ratio(contact_flags: Sequence[bool]) -> float:
    values = list(contact_flags)
    if not values:
        return 0.0
    return float(sum(1 for value in values if value) / len(values))


def compute_red_motion_after_close(
    final_displacement_from_close_start: float,
    close_start_displacement: float = 0.0,
) -> float:
    return float(final_displacement_from_close_start - close_start_displacement)


def choose_best_trial(trials: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    if not trials:
        return None
    return max(
        trials,
        key=lambda item: (
            float(item.get(metric, 0.0)),
            float(item.get("post_close_max_force", 0.0)),
            -float(item.get("overall_min_tip_red_distance", float("inf"))),
        ),
    )


def make_judgment(
    *,
    any_post_close_contact: bool,
    max_post_close_contact_ratio: float,
    max_red_motion_after_close: float,
    any_intermittent_contact: bool,
) -> str:
    if float(max_post_close_contact_ratio) >= 0.25:
        return "[OK] contact persisted during hold/retract diagnostic"
    if any_post_close_contact:
        return "[WARN] contact occurred but persistence remains weak"
    if any_intermittent_contact and float(max_red_motion_after_close) > 0.005:
        return "[WARN] object moved but contact was intermittent"
    return "[FAIL] no useful post-close contact observed"


def _record_trial_displacement(record: dict[str, Any]) -> float:
    if "red_displacement_from_trial_start" in record:
        return float(record["red_displacement_from_trial_start"])
    return float(record.get("red_displacement", 0.0))


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
    red_displacement_start = _record_trial_displacement(phase_records[0])
    red_displacement_end = _record_trial_displacement(phase_records[-1])

    return {
        "phase": phase_name,
        "steps": int(len(phase_records)),
        "min_tip_red_distance": float(min(record["tip_red_distance"] for record in phase_records)),
        "final_tip_red_distance": float(phase_records[-1]["tip_red_distance"]),
        "has_red_grasper_contact": bool(contact_records),
        "contact_step_count": int(len(contact_records)),
        "contact_ratio": compute_post_close_contact_ratio(
            [bool(record["has_red_grasper_contact"]) for record in phase_records]
        ),
        "max_red_grasper_normal_force": float(max(contact_forces)) if contact_forces else 0.0,
        "mean_red_grasper_normal_force": float(sum(contact_forces) / len(contact_forces)) if contact_forces else 0.0,
        "red_displacement_start": red_displacement_start,
        "red_displacement_end": red_displacement_end,
        "red_displacement_delta": float(red_displacement_end - red_displacement_start),
    }


def _distance(position_a: Sequence[float], position_b: Sequence[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(position_a, dtype=np.float64) - np.asarray(position_b, dtype=np.float64)
        )
    )


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
        labels = [
            str(contact["geom1"]),
            str(contact["geom2"]),
            str(contact["body1"]),
            str(contact["body2"]),
        ]
        lowered = [label.lower() for label in labels]
        has_red = any("red_object" in label for label in lowered)
        has_pedestal = any("red_pedestal" in label for label in lowered)
        has_table = any("tabletop" in label or "table" in label for label in lowered)
        has_grasper = any(
            needle in label
            for label in lowered
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


def _lerp(start_value: float, end_value: float, step_index: int, total_steps: int) -> float:
    if total_steps <= 0:
        return float(end_value)
    alpha = float(step_index + 1) / float(total_steps)
    return float((1.0 - alpha) * float(start_value) + alpha * float(end_value))


def _make_action(section_scale: float, grip_command: float) -> dict[str, Any]:
    return {
        "section_angles": scale_section_angles(FULL_SECTION_ANGLES, scale=section_scale),
        "grip_command": float(grip_command),
        "grasper_rotation": float(GRASPER_ROTATION),
    }


def _record_step(
    *,
    trial_name: str,
    phase_name: str,
    global_step: int,
    section_scale: float,
    grip_command: float,
    mujoco: Any,
    model: Any,
    data: Any,
    trial_start_red_position: Sequence[float],
    close_start_red_position: Sequence[float] | None,
) -> dict[str, Any]:
    tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    red_position = _body_position(mujoco, model, data, "red_object")
    if tip_position is None or red_position is None:
        raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")
    contacts = _contact_snapshot(mujoco, model, data)
    flags = _classify_contacts(contacts)
    return {
        "trial": trial_name,
        "phase": phase_name,
        "global_step": int(global_step),
        "tip_position": [float(value) for value in tip_position],
        "red_position": [float(value) for value in red_position],
        "red_displacement_from_trial_start": float(_distance(trial_start_red_position, red_position)),
        "red_displacement_from_close_start": 0.0
        if close_start_red_position is None
        else float(_distance(close_start_red_position, red_position)),
        "tip_red_distance": float(_distance(tip_position, red_position)),
        "grip_command": float(grip_command),
        "section_scale": float(section_scale),
        "grasper_rotation": float(GRASPER_ROTATION),
        "contact_pairs": sorted(
            {" <-> ".join(sorted([str(contact["geom1"]), str(contact["geom2"])])) for contact in contacts}
        ),
        "has_red_grasper_contact": bool(flags["has_red_grasper_contact"]),
        "has_red_pedestal_contact": bool(flags["has_red_pedestal_contact"]),
        "has_red_table_contact": bool(flags["has_red_table_contact"]),
        "red_grasper_normal_force": float(flags["red_grasper_normal_force"]),
        "num_contacts": int(data.ncon),
    }


def _run_trial(
    schedule: dict[str, Any],
    *,
    model: Any,
    data: Any,
    mujoco: Any,
    settle_steps: int,
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

    initial_red_position = _body_position(mujoco, model, data, "red_object")
    initial_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    if initial_red_position is None or initial_tip_position is None:
        raise RuntimeError("Missing red_object or feagine_grasper_tip body pose.")

    records: list[dict[str, Any]] = []
    global_step = 0
    close_start_red_position: list[float] | None = None
    phase_summaries: list[dict[str, Any]] = []

    for phase in schedule["phases"]:
        phase_name = str(phase["name"])
        steps = int(phase["steps"])
        if phase_name == "close_ramp":
            close_start_red_position = _body_position(mujoco, model, data, "red_object")
            if close_start_red_position is None:
                raise RuntimeError("Missing red_object body pose at close start.")

        for step_index in range(steps):
            section_scale = _lerp(phase["scale_start"], phase["scale_end"], step_index, steps)
            grip_command = _lerp(phase["grip_start"], phase["grip_end"], step_index, steps)
            action = _make_action(section_scale=section_scale, grip_command=grip_command)
            _apply_feagine_action(robot, action)
            mujoco.mj_step(model, data)
            records.append(
                _record_step(
                    trial_name=str(schedule["name"]),
                    phase_name=phase_name,
                    global_step=global_step,
                    section_scale=section_scale,
                    grip_command=grip_command,
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    trial_start_red_position=initial_red_position,
                    close_start_red_position=close_start_red_position,
                )
            )
            global_step += 1

        phase_summaries.append(summarize_phase(phase_name, records))

    final_red_position = _body_position(mujoco, model, data, "red_object")
    final_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    if final_red_position is None or final_tip_position is None:
        raise RuntimeError("Missing red_object or feagine_grasper_tip body pose.")

    contact_records = [record for record in records if bool(record["has_red_grasper_contact"])]
    contact_first_phase = None if not contact_records else str(contact_records[0]["phase"])
    contact_last_phase = None if not contact_records else str(contact_records[-1]["phase"])
    post_close_phase_names = {
        "hold_closed_initial",
        *(str(phase["name"]) for phase in schedule["phases"][4:]),
    }
    post_close_records = [record for record in records if record["phase"] in post_close_phase_names]
    post_close_contact_steps = [bool(record["has_red_grasper_contact"]) for record in post_close_records]
    post_close_contact_ratio = compute_post_close_contact_ratio(post_close_contact_steps)
    post_close_contact_forces = [
        float(record["red_grasper_normal_force"])
        for record in post_close_records
        if bool(record["has_red_grasper_contact"])
    ]
    red_motion_after_close = 0.0 if not post_close_records else float(
        max(record["red_displacement_from_close_start"] for record in post_close_records)
    )

    summary = {
        "name": str(schedule["name"]),
        "initial_red_position": [float(value) for value in initial_red_position],
        "final_red_position": [float(value) for value in final_red_position],
        "initial_tip_position": [float(value) for value in initial_tip_position],
        "final_tip_position": [float(value) for value in final_tip_position],
        "overall_min_tip_red_distance": float(min(record["tip_red_distance"] for record in records)),
        "overall_max_red_displacement": float(
            max(record["red_displacement_from_trial_start"] for record in records)
        ),
        "total_red_displacement": float(_distance(initial_red_position, final_red_position)),
        "contact_first_phase": contact_first_phase,
        "contact_last_phase": contact_last_phase,
        "post_close_contact_ratio": float(post_close_contact_ratio),
        "post_close_max_force": float(max(post_close_contact_forces)) if post_close_contact_forces else 0.0,
        "post_close_mean_force": float(sum(post_close_contact_forces) / len(post_close_contact_forces))
        if post_close_contact_forces
        else 0.0,
        "red_motion_after_close": float(compute_red_motion_after_close(red_motion_after_close)),
        "phase_summaries": phase_summaries,
        "contact_pairs_seen": sorted({pair for record in records for pair in record["contact_pairs"]}),
        "any_contact_any_phase": bool(contact_records),
        "max_red_grasper_normal_force": float(
            max(float(record["red_grasper_normal_force"]) for record in records)
        ),
    }
    if save_trajectory:
        summary["trajectory"] = records
    return summary


def run_hold_retract_diagnostic(
    scene_path: str | Path,
    *,
    settle_steps: int,
    save_trajectory: bool,
) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model, data = _load_tabletop_scene(str(resolved_scene))

    try:
        _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc

    schedules = build_trial_schedules()
    trial_summaries = [
        _run_trial(
            schedule,
            model=model,
            data=data,
            mujoco=mujoco,
            settle_steps=settle_steps,
            save_trajectory=save_trajectory,
        )
        for schedule in schedules
    ]

    best_trial_by_contact_ratio = choose_best_trial(trial_summaries, "post_close_contact_ratio")
    best_trial_by_red_motion = choose_best_trial(trial_summaries, "red_motion_after_close")
    any_post_close_contact = any(float(trial["post_close_contact_ratio"]) > 0.0 for trial in trial_summaries)
    max_post_close_contact_ratio = max(
        float(trial["post_close_contact_ratio"]) for trial in trial_summaries
    )
    max_red_motion_after_close = max(float(trial["red_motion_after_close"]) for trial in trial_summaries)
    max_red_grasper_normal_force = max(
        float(trial["max_red_grasper_normal_force"]) for trial in trial_summaries
    )
    any_intermittent_contact = any(bool(trial["any_contact_any_phase"]) for trial in trial_summaries)
    judgment = make_judgment(
        any_post_close_contact=any_post_close_contact,
        max_post_close_contact_ratio=max_post_close_contact_ratio,
        max_red_motion_after_close=max_red_motion_after_close,
        any_intermittent_contact=any_intermittent_contact,
    )

    return {
        "scene": str(resolved_scene),
        "target": {
            "full_section_angles": [float(value) for value in FULL_SECTION_ANGLES],
            "preclose_scale": float(PRECLOSE_SCALE),
            "preclose_section_angles": scale_section_angles(FULL_SECTION_ANGLES, scale=PRECLOSE_SCALE),
            "grasper_rotation": float(GRASPER_ROTATION),
        },
        "trials": trial_summaries,
        "best_trial_by_contact_ratio": best_trial_by_contact_ratio,
        "best_trial_by_red_motion": best_trial_by_red_motion,
        "overall": {
            "any_post_close_contact": bool(any_post_close_contact),
            "max_post_close_contact_ratio": float(max_post_close_contact_ratio),
            "max_red_motion_after_close": float(max_red_motion_after_close),
            "max_red_grasper_normal_force": float(max_red_grasper_normal_force),
        },
        "judgment": judgment,
        "notes": [
            "This is a hold/retract contact persistence diagnostic.",
            "It does not claim stable grasp or object lifting.",
        ],
    }


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
        report = run_hold_retract_diagnostic(
            scene_path,
            settle_steps=args.settle_steps,
            save_trajectory=bool(args.save_trajectory),
        )
    except ValueError as exc:
        if "unsupported action keys" in str(exc):
            print(f"[FAIL] {exc}")
            return 4
        print(f"[FAIL] hold/retract diagnostic failed: {exc}")
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
        print(f"[FAIL] hold/retract diagnostic failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] hold/retract diagnostic failed: {exc}")
        return 1

    for trial in report["trials"]:
        print(
            f"[TRIAL] {trial['name']} "
            f"post_close_contact_ratio={trial['post_close_contact_ratio']} "
            f"red_motion_after_close={trial['red_motion_after_close']} "
            f"max_force={trial['post_close_max_force']}"
        )

    best_trial_by_contact_ratio = report["best_trial_by_contact_ratio"]
    best_trial_by_red_motion = report["best_trial_by_red_motion"]
    print(
        "[SUMMARY] best_trial_by_contact_ratio="
        f"{None if best_trial_by_contact_ratio is None else best_trial_by_contact_ratio['name']}"
    )
    print(
        "[SUMMARY] best_trial_by_red_motion="
        f"{None if best_trial_by_red_motion is None else best_trial_by_red_motion['name']}"
    )
    print(f"[SUMMARY] max_post_close_contact_ratio={report['overall']['max_post_close_contact_ratio']}")
    print(f"[SUMMARY] max_red_motion_after_close={report['overall']['max_red_motion_after_close']}")
    print(f"[SUMMARY] judgment={report['judgment']}")

    try:
        saved_path = write_report(report, output_path)
    except Exception as exc:
        print(f"[FAIL] failed to write diagnostics JSON: {exc}")
        return 1

    print(f"[OK] wrote {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
