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
BASE_SCALE = 0.74
APPROACH_ROTATION = -0.7854
CLOSE_ROTATION = -0.7854
CLOSE_ADVANCE_SCALE = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-close retract/lift-like following diagnostics."
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
            "scale_end": BASE_SCALE,
            "grip_start": 0.0,
            "grip_end": 0.0,
            "rotation_start": 0.0,
            "rotation_end": APPROACH_ROTATION,
        },
        {
            "name": "hold_preclose",
            "steps": 160,
            "scale_start": BASE_SCALE,
            "scale_end": BASE_SCALE,
            "grip_start": 0.0,
            "grip_end": 0.0,
            "rotation_start": APPROACH_ROTATION,
            "rotation_end": APPROACH_ROTATION,
        },
        {
            "name": "close_ramp",
            "steps": 120,
            "scale_start": BASE_SCALE,
            "scale_end": BASE_SCALE + CLOSE_ADVANCE_SCALE,
            "grip_start": 0.0,
            "grip_end": 1.0,
            "rotation_start": APPROACH_ROTATION,
            "rotation_end": CLOSE_ROTATION,
        },
        {
            "name": "hold_closed_initial",
            "steps": 240,
            "scale_start": BASE_SCALE + CLOSE_ADVANCE_SCALE,
            "scale_end": BASE_SCALE + CLOSE_ADVANCE_SCALE,
            "grip_start": 1.0,
            "grip_end": 1.0,
            "rotation_start": CLOSE_ROTATION,
            "rotation_end": CLOSE_ROTATION,
        },
    ]
    return [
        {
            "name": "hold_only",
            "phases": shared_phases
            + [
                {
                    "name": "post_hold",
                    "steps": 600,
                    "scale_start": BASE_SCALE,
                    "scale_end": BASE_SCALE,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": CLOSE_ROTATION,
                    "rotation_end": CLOSE_ROTATION,
                }
            ],
        },
        {
            "name": "gentle_retract_scale_down",
            "phases": shared_phases
            + [
                {
                    "name": "retract_ramp",
                    "steps": 180,
                    "scale_start": 0.74,
                    "scale_end": 0.66,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": CLOSE_ROTATION,
                    "rotation_end": CLOSE_ROTATION,
                },
                {
                    "name": "retract_hold",
                    "steps": 360,
                    "scale_start": 0.66,
                    "scale_end": 0.66,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": CLOSE_ROTATION,
                    "rotation_end": CLOSE_ROTATION,
                },
            ],
        },
        {
            "name": "gentle_advance_scale_up",
            "phases": shared_phases
            + [
                {
                    "name": "advance_ramp",
                    "steps": 180,
                    "scale_start": 0.74,
                    "scale_end": 0.80,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": CLOSE_ROTATION,
                    "rotation_end": CLOSE_ROTATION,
                },
                {
                    "name": "advance_hold",
                    "steps": 360,
                    "scale_start": 0.80,
                    "scale_end": 0.80,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": CLOSE_ROTATION,
                    "rotation_end": CLOSE_ROTATION,
                },
            ],
        },
        {
            "name": "rotation_settle",
            "phases": shared_phases
            + [
                {
                    "name": "rotation_ramp",
                    "steps": 180,
                    "scale_start": BASE_SCALE,
                    "scale_end": BASE_SCALE,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": CLOSE_ROTATION,
                    "rotation_end": -1.1781,
                },
                {
                    "name": "rotation_hold",
                    "steps": 360,
                    "scale_start": BASE_SCALE,
                    "scale_end": BASE_SCALE,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": -1.1781,
                    "rotation_end": -1.1781,
                },
            ],
        },
        {
            "name": "micro_lift_like",
            "phases": shared_phases
            + [
                {
                    "name": "lift_like_ramp",
                    "steps": 180,
                    "scale_start": 0.74,
                    "scale_end": 0.68,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": CLOSE_ROTATION,
                    "rotation_end": -1.1781,
                },
                {
                    "name": "lift_like_hold",
                    "steps": 360,
                    "scale_start": 0.68,
                    "scale_end": 0.68,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": -1.1781,
                    "rotation_end": -1.1781,
                },
            ],
        },
    ]


def compute_contact_ratio(contact_flags: Sequence[bool]) -> float:
    values = list(contact_flags)
    if not values:
        return 0.0
    return float(sum(1 for value in values if value) / len(values))


def compute_post_close_contact_ratio(contact_flags: Sequence[bool]) -> float:
    return compute_contact_ratio(contact_flags)


def _extract_z(value: float | Sequence[float]) -> float:
    if isinstance(value, Sequence) and not isinstance(value, str):
        values = _as_float_sequence(value, "position")
        if len(values) < 3:
            raise ValueError("position must contain at least 3 values.")
        return float(values[2])
    return float(value)


def compute_red_z_delta(red_z_start: float | Sequence[float], red_z_end: float | Sequence[float]) -> float:
    return float(_extract_z(red_z_end) - _extract_z(red_z_start))


def compute_object_follow_score(
    *,
    post_close_red_grasper_contact_ratio: float,
    post_close_red_motion: float,
    post_close_red_z_delta: float,
    post_close_red_pedestal_contact_ratio: float,
    post_close_max_grasper_force: float,
) -> float:
    score = 0.0
    score += float(post_close_red_grasper_contact_ratio) * 3.0
    score += min(float(post_close_red_motion) / 0.03, 1.0) * 2.0
    if float(post_close_red_z_delta) > 0.003:
        score += 1.0
    if float(post_close_red_pedestal_contact_ratio) < 0.5:
        score += 1.0
    if float(post_close_max_grasper_force) > 500.0:
        score -= 1.5
    if float(post_close_max_grasper_force) > 1000.0:
        score -= 2.0
    return float(score)


def select_best_trial(trials: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    if not trials:
        return None
    return max(
        trials,
        key=lambda item: (
            float(item.get(metric, 0.0)),
            float(item.get("post_close_red_grasper_contact_ratio", 0.0)),
            -float(item.get("post_close_max_grasper_force", float("inf"))),
        ),
    )


def choose_best_trial(trials: Sequence[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    return select_best_trial(trials, metric)


def make_judgment(*, trials: Sequence[dict[str, Any]]) -> str:
    if any(
        float(trial.get("post_close_red_grasper_contact_ratio", 0.0)) >= 0.5
        and float(trial.get("post_close_red_motion", 0.0)) >= 0.01
        and float(trial.get("post_close_max_grasper_force", 0.0)) < 1000.0
        for trial in trials
    ):
        return "[OK] object followed grasper during post-close diagnostic"
    if any(
        float(trial.get("post_close_red_grasper_contact_ratio", 0.0)) >= 0.5
        and float(trial.get("post_close_red_motion", 0.0)) < 0.01
        for trial in trials
    ):
        return "[WARN] contact persisted but object following is weak"
    if any(
        float(trial.get("post_close_red_motion", 0.0)) >= 0.01
        and float(trial.get("post_close_max_grasper_force", 0.0)) >= 1000.0
        for trial in trials
    ):
        return "[WARN] object motion observed but force is extreme"
    return "[FAIL] no useful post-close following observed"


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
    red_pedestal_forces: list[float] = []

    for contact in contacts:
        labels = [
            str(contact["geom1"]),
            str(contact["geom2"]),
            str(contact["body1"]),
            str(contact["body2"]),
        ]
        lowered = [label.lower() for label in labels]
        normal_force = abs(float(contact["normal_force"]))
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
            red_grasper_forces.append(normal_force)
        if has_red and has_pedestal:
            red_pedestal_forces.append(normal_force)

    return {
        "has_red_grasper_contact": bool(has_red_grasper_contact),
        "has_red_pedestal_contact": bool(has_red_pedestal_contact),
        "has_red_table_contact": bool(has_red_table_contact),
        "red_grasper_normal_force": float(max(red_grasper_forces)) if red_grasper_forces else 0.0,
        "red_pedestal_normal_force": float(max(red_pedestal_forces)) if red_pedestal_forces else 0.0,
    }


def _lerp(start_value: float, end_value: float, step_index: int, total_steps: int) -> float:
    if total_steps <= 0:
        return float(end_value)
    alpha = float(step_index + 1) / float(total_steps)
    return float((1.0 - alpha) * float(start_value) + alpha * float(end_value))


def _interpolate_section_angles(
    start_angles: Sequence[float],
    end_angles: Sequence[float],
    *,
    step_index: int,
    total_steps: int,
) -> list[float]:
    if total_steps <= 0:
        return [float(value) for value in end_angles]
    alpha = float(step_index + 1) / float(total_steps)
    return [
        float((1.0 - alpha) * float(start_value) + alpha * float(end_value))
        for start_value, end_value in zip(start_angles, end_angles)
    ]


def summarize_phase(phase_name: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    phase_records = [record for record in records if record["phase"] == phase_name]
    if not phase_records:
        return {
            "phase": phase_name,
            "steps": 0,
            "min_tip_red_distance": None,
            "final_tip_red_distance": None,
            "red_z_start": 0.0,
            "red_z_end": 0.0,
            "red_z_delta": 0.0,
            "has_red_grasper_contact": False,
            "red_grasper_contact_ratio": 0.0,
            "red_pedestal_contact_ratio": 0.0,
            "red_table_contact_ratio": 0.0,
            "max_red_grasper_normal_force": 0.0,
            "mean_red_grasper_normal_force": 0.0,
            "max_red_pedestal_normal_force": 0.0,
            "red_displacement_start": 0.0,
            "red_displacement_end": 0.0,
            "red_displacement_delta": 0.0,
        }

    contact_records = [record for record in phase_records if bool(record["has_red_grasper_contact"])]
    grasper_forces = [float(record["red_grasper_normal_force"]) for record in contact_records]
    pedestal_forces = [float(record["red_pedestal_normal_force"]) for record in phase_records]
    red_z_start = float(phase_records[0]["red_z"])
    red_z_end = float(phase_records[-1]["red_z"])
    red_displacement_start = float(phase_records[0]["red_displacement_from_trial_start"])
    red_displacement_end = float(phase_records[-1]["red_displacement_from_trial_start"])

    return {
        "phase": phase_name,
        "steps": int(len(phase_records)),
        "min_tip_red_distance": float(min(record["tip_red_distance"] for record in phase_records)),
        "final_tip_red_distance": float(phase_records[-1]["tip_red_distance"]),
        "red_z_start": red_z_start,
        "red_z_end": red_z_end,
        "red_z_delta": float(compute_red_z_delta(red_z_start, red_z_end)),
        "has_red_grasper_contact": bool(contact_records),
        "red_grasper_contact_ratio": float(
            sum(1 for record in phase_records if bool(record["has_red_grasper_contact"])) / len(phase_records)
        ),
        "red_pedestal_contact_ratio": float(
            sum(1 for record in phase_records if bool(record["has_red_pedestal_contact"])) / len(phase_records)
        ),
        "red_table_contact_ratio": float(
            sum(1 for record in phase_records if bool(record["has_red_table_contact"])) / len(phase_records)
        ),
        "max_red_grasper_normal_force": float(max(grasper_forces)) if grasper_forces else 0.0,
        "mean_red_grasper_normal_force": float(sum(grasper_forces) / len(grasper_forces))
        if grasper_forces
        else 0.0,
        "max_red_pedestal_normal_force": float(max(pedestal_forces)) if pedestal_forces else 0.0,
        "red_displacement_start": red_displacement_start,
        "red_displacement_end": red_displacement_end,
        "red_displacement_delta": float(red_displacement_end - red_displacement_start),
    }


def _record_step(
    *,
    trial_name: str,
    phase_name: str,
    global_step: int,
    section_scale: float,
    grip_command: float,
    grasper_rotation: float,
    mujoco: Any,
    model: Any,
    data: Any,
    trial_start_red_position: Sequence[float],
    post_close_start_red_position: Sequence[float] | None,
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
        "red_z": float(red_position[2]),
        "red_displacement_from_trial_start": float(_distance(trial_start_red_position, red_position)),
        "red_displacement_from_post_close_start": 0.0
        if post_close_start_red_position is None
        else float(_distance(post_close_start_red_position, red_position)),
        "tip_red_distance": float(_distance(tip_position, red_position)),
        "grip_command": float(grip_command),
        "section_scale": float(section_scale),
        "grasper_rotation": float(grasper_rotation),
        "contact_pairs": sorted(
            {" <-> ".join(sorted([str(contact["geom1"]), str(contact["geom2"])])) for contact in contacts}
        ),
        "has_red_grasper_contact": bool(flags["has_red_grasper_contact"]),
        "has_red_pedestal_contact": bool(flags["has_red_pedestal_contact"]),
        "has_red_table_contact": bool(flags["has_red_table_contact"]),
        "red_grasper_normal_force": float(flags["red_grasper_normal_force"]),
        "red_pedestal_normal_force": float(flags["red_pedestal_normal_force"]),
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
    phase_summaries: dict[str, Any] = {}
    global_step = 0
    post_close_start_red_position: list[float] | None = None

    for phase in schedule["phases"]:
        phase_name = str(phase["name"])
        steps = int(phase["steps"])
        if phase_name == "hold_closed_initial":
            post_close_start_red_position = _body_position(mujoco, model, data, "red_object")
            if post_close_start_red_position is None:
                raise RuntimeError("Missing red_object body pose at post-close start.")

        start_section = scale_section_angles(FULL_SECTION_ANGLES, scale=float(phase["scale_start"]))
        end_section = scale_section_angles(FULL_SECTION_ANGLES, scale=float(phase["scale_end"]))

        for step_index in range(steps):
            action = {
                "section_angles": _interpolate_section_angles(
                    start_section,
                    end_section,
                    step_index=step_index,
                    total_steps=steps,
                ),
                "grip_command": _lerp(float(phase["grip_start"]), float(phase["grip_end"]), step_index, steps),
                "grasper_rotation": _lerp(
                    float(phase["rotation_start"]),
                    float(phase["rotation_end"]),
                    step_index,
                    steps,
                ),
            }
            _apply_feagine_action(robot, action)
            mujoco.mj_step(model, data)
            records.append(
                _record_step(
                    trial_name=str(schedule["name"]),
                    phase_name=phase_name,
                    global_step=global_step,
                    section_scale=_lerp(float(phase["scale_start"]), float(phase["scale_end"]), step_index, steps),
                    grip_command=float(action["grip_command"]),
                    grasper_rotation=float(action["grasper_rotation"]),
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    trial_start_red_position=initial_red_position,
                    post_close_start_red_position=post_close_start_red_position,
                )
            )
            global_step += 1

        phase_summaries[phase_name] = summarize_phase(phase_name, records)

    final_red_position = _body_position(mujoco, model, data, "red_object")
    final_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    if final_red_position is None or final_tip_position is None or post_close_start_red_position is None:
        raise RuntimeError("Missing final red/tip/post-close poses.")

    post_close_phase_names = [phase["name"] for phase in schedule["phases"][3:]]
    post_close_records = [record for record in records if record["phase"] in post_close_phase_names]
    contact_pairs_seen = sorted({pair for record in records for pair in record["contact_pairs"]})
    post_close_grasper_contact_ratio = compute_post_close_contact_ratio(
        [bool(record["has_red_grasper_contact"]) for record in post_close_records]
    )
    post_close_pedestal_contact_ratio = compute_post_close_contact_ratio(
        [bool(record["has_red_pedestal_contact"]) for record in post_close_records]
    )
    post_close_grasper_forces = [
        float(record["red_grasper_normal_force"])
        for record in post_close_records
        if bool(record["has_red_grasper_contact"])
    ]
    post_close_pedestal_forces = [float(record["red_pedestal_normal_force"]) for record in post_close_records]
    post_close_red_motion = max(float(record["red_displacement_from_post_close_start"]) for record in post_close_records)
    post_close_red_z_start = float(post_close_records[0]["red_z"])
    post_close_red_z_end = float(post_close_records[-1]["red_z"])
    post_close_red_z_delta = compute_red_z_delta(post_close_red_z_start, post_close_red_z_end)
    post_close_max_grasper_force = max(post_close_grasper_forces) if post_close_grasper_forces else 0.0
    post_close_mean_grasper_force = (
        sum(post_close_grasper_forces) / len(post_close_grasper_forces)
        if post_close_grasper_forces
        else 0.0
    )
    post_close_max_pedestal_force = max(post_close_pedestal_forces) if post_close_pedestal_forces else 0.0
    force_extreme = bool(post_close_max_grasper_force >= 1000.0)
    object_follow_score = compute_object_follow_score(
        post_close_red_grasper_contact_ratio=post_close_grasper_contact_ratio,
        post_close_red_motion=post_close_red_motion,
        post_close_red_z_delta=post_close_red_z_delta,
        post_close_red_pedestal_contact_ratio=post_close_pedestal_contact_ratio,
        post_close_max_grasper_force=post_close_max_grasper_force,
    )

    summary = {
        "name": str(schedule["name"]),
        "initial_red_position": [float(value) for value in initial_red_position],
        "final_red_position": [float(value) for value in final_red_position],
        "post_close_start_red_position": [float(value) for value in post_close_start_red_position],
        "initial_tip_position": [float(value) for value in initial_tip_position],
        "final_tip_position": [float(value) for value in final_tip_position],
        "overall_min_tip_red_distance": float(min(record["tip_red_distance"] for record in records)),
        "overall_max_red_displacement": float(
            max(record["red_displacement_from_trial_start"] for record in records)
        ),
        "total_red_displacement": float(_distance(initial_red_position, final_red_position)),
        "post_close_red_motion": float(post_close_red_motion),
        "post_close_red_z_delta": float(post_close_red_z_delta),
        "post_close_red_grasper_contact_ratio": float(post_close_grasper_contact_ratio),
        "post_close_red_pedestal_contact_ratio": float(post_close_pedestal_contact_ratio),
        "post_close_max_grasper_force": float(post_close_max_grasper_force),
        "post_close_mean_grasper_force": float(post_close_mean_grasper_force),
        "post_close_max_pedestal_force": float(post_close_max_pedestal_force),
        "force_extreme": bool(force_extreme),
        "object_follow_score": float(object_follow_score),
        "phase_summaries": phase_summaries,
        "contact_pairs_seen": contact_pairs_seen,
    }
    if save_trajectory:
        summary["trajectory"] = records
    return summary


def run_retract_lift_diagnostic(
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
    trials = [
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

    best_trial_by_follow_score = choose_best_trial(trials, "object_follow_score")
    best_trial_by_z_delta = choose_best_trial(trials, "post_close_red_z_delta")
    best_trial_by_contact_ratio = choose_best_trial(trials, "post_close_red_grasper_contact_ratio")
    max_post_close_contact_ratio = max(float(trial["post_close_red_grasper_contact_ratio"]) for trial in trials)
    max_post_close_red_motion = max(float(trial["post_close_red_motion"]) for trial in trials)
    max_post_close_red_z_delta = max(float(trial["post_close_red_z_delta"]) for trial in trials)
    min_post_close_pedestal_contact_ratio = min(
        float(trial["post_close_red_pedestal_contact_ratio"]) for trial in trials
    )
    max_red_grasper_normal_force = max(float(trial["post_close_max_grasper_force"]) for trial in trials)

    judgment = make_judgment(trials=trials)

    return {
        "scene": str(resolved_scene),
        "target": {
            "full_section_angles": [float(value) for value in FULL_SECTION_ANGLES],
            "base_scale": float(BASE_SCALE),
            "approach_rotation": float(APPROACH_ROTATION),
            "close_rotation": float(CLOSE_ROTATION),
            "close_advance_scale": float(CLOSE_ADVANCE_SCALE),
            "base_section_angles": scale_section_angles(FULL_SECTION_ANGLES, scale=BASE_SCALE),
        },
        "trials": trials,
        "best_trial_by_follow_score": best_trial_by_follow_score,
        "best_trial_by_z_delta": best_trial_by_z_delta,
        "best_trial_by_contact_ratio": best_trial_by_contact_ratio,
        "overall": {
            "any_post_close_red_grasper_contact": any(
                float(trial["post_close_red_grasper_contact_ratio"]) > 0.0 for trial in trials
            ),
            "max_post_close_contact_ratio": float(max_post_close_contact_ratio),
            "max_post_close_red_motion": float(max_post_close_red_motion),
            "max_post_close_red_z_delta": float(max_post_close_red_z_delta),
            "min_post_close_pedestal_contact_ratio": float(min_post_close_pedestal_contact_ratio),
            "max_red_grasper_normal_force": float(max_red_grasper_normal_force),
        },
        "judgment": judgment,
        "notes": [
            "This is a post-close retract/lift-like following diagnostic.",
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
        report = run_retract_lift_diagnostic(
            scene_path,
            settle_steps=args.settle_steps,
            save_trajectory=bool(args.save_trajectory),
        )
    except ValueError as exc:
        if "unsupported action keys" in str(exc):
            print(f"[FAIL] {exc}")
            return 4
        print(f"[FAIL] retract/lift diagnostic failed: {exc}")
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
        print(f"[FAIL] retract/lift diagnostic failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] retract/lift diagnostic failed: {exc}")
        return 1

    for trial in report["trials"]:
        print(f"[TRIAL] {trial['name']} ...")
        print(
            f"[RESULT] {trial['name']} "
            f"contact_ratio={trial['post_close_red_grasper_contact_ratio']} "
            f"red_motion={trial['post_close_red_motion']} "
            f"z_delta={trial['post_close_red_z_delta']} "
            f"max_force={trial['post_close_max_grasper_force']} "
            f"score={trial['object_follow_score']}"
        )

    print(
        "[SUMMARY] best_trial_by_follow_score="
        f"{None if report['best_trial_by_follow_score'] is None else report['best_trial_by_follow_score']['name']}"
    )
    print(
        "[SUMMARY] best_trial_by_z_delta="
        f"{None if report['best_trial_by_z_delta'] is None else report['best_trial_by_z_delta']['name']}"
    )
    print(
        "[SUMMARY] best_trial_by_contact_ratio="
        f"{None if report['best_trial_by_contact_ratio'] is None else report['best_trial_by_contact_ratio']['name']}"
    )
    print(f"[SUMMARY] max_post_close_contact_ratio={report['overall']['max_post_close_contact_ratio']}")
    print(f"[SUMMARY] max_post_close_red_motion={report['overall']['max_post_close_red_motion']}")
    print(f"[SUMMARY] max_post_close_red_z_delta={report['overall']['max_post_close_red_z_delta']}")
    print(
        "[SUMMARY] min_post_close_pedestal_contact_ratio="
        f"{report['overall']['min_post_close_pedestal_contact_ratio']}"
    )
    print(f"[SUMMARY] max_red_grasper_normal_force={report['overall']['max_red_grasper_normal_force']}")
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
