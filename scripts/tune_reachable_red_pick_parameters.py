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
PRECLOSE_SCALES = [0.58, 0.62, 0.66, 0.70, 0.74]
APPROACH_ROTATIONS = [0.0, -0.3927, -0.7854]
CLOSE_ROTATIONS = [-0.3927, -0.7854, -1.1781]
CLOSE_ADVANCE_SCALES = [0.0, 0.03, 0.06]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune reachable red-object parameters for clean preclose and persistent post-close contact."
    )
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--output", required=True, help="Path to the diagnostics JSON.")
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--approach-ramp-steps", type=int, default=120)
    parser.add_argument("--hold-preclose-steps", type=int, default=160)
    parser.add_argument("--close-ramp-steps", type=int, default=120)
    parser.add_argument("--hold-closed-steps", type=int, default=300)
    parser.add_argument("--max-trials", type=int, default=135)
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


def build_trial_grid(
    *,
    preclose_scales: Sequence[float] = PRECLOSE_SCALES,
    approach_rotations: Sequence[float] = APPROACH_ROTATIONS,
    close_rotations: Sequence[float] = CLOSE_ROTATIONS,
    close_advance_scales: Sequence[float] = CLOSE_ADVANCE_SCALES,
    max_trials: int = 135,
) -> list[dict[str, float]]:
    grid: list[dict[str, float]] = []
    for preclose_scale in preclose_scales:
        for approach_rotation in approach_rotations:
            for close_rotation in close_rotations:
                for close_advance_scale in close_advance_scales:
                    final_close_scale = min(1.0, float(preclose_scale) + float(close_advance_scale))
                    grid.append(
                        {
                            "preclose_scale": float(preclose_scale),
                            "approach_rotation": float(approach_rotation),
                            "close_rotation": float(close_rotation),
                            "close_advance_scale": float(close_advance_scale),
                            "final_close_scale": float(final_close_scale),
                        }
                    )
                    if len(grid) >= int(max_trials):
                        return grid
    return grid


def compute_trial_flags(
    *,
    hold_preclose_contact_ratio: float,
    close_ramp_contact_ratio: float,
    hold_closed_contact_ratio: float,
    red_motion_after_close: float,
    max_red_grasper_normal_force: float,
) -> dict[str, bool]:
    clean_preclose = float(hold_preclose_contact_ratio) <= 0.02
    close_added_contact = float(close_ramp_contact_ratio) > 0.0 or float(hold_closed_contact_ratio) > 0.0
    persistent_post_close = float(hold_closed_contact_ratio) >= 0.25
    red_object_moved_after_close = float(red_motion_after_close) > 0.005
    force_not_extreme = float(max_red_grasper_normal_force) < 500.0
    return {
        "clean_preclose": bool(clean_preclose),
        "close_added_contact": bool(close_added_contact),
        "persistent_post_close": bool(persistent_post_close),
        "red_object_moved_after_close": bool(red_object_moved_after_close),
        "force_not_extreme": bool(force_not_extreme),
    }


def compute_trial_score(
    *,
    clean_preclose: bool,
    close_added_contact: bool,
    persistent_post_close: bool,
    red_object_moved_after_close: bool,
    force_not_extreme: bool,
    min_tip_red_distance: float,
    max_red_grasper_normal_force: float,
    hold_preclose_contact_ratio: float,
) -> float:
    score = 0.0
    if clean_preclose:
        score += 3.0
    if close_added_contact:
        score += 3.0
    if persistent_post_close:
        score += 2.0
    if red_object_moved_after_close:
        score += 1.0
    if force_not_extreme:
        score += 1.0

    score += max(0.0, 1.0 - float(min_tip_red_distance) / 0.20)

    if float(max_red_grasper_normal_force) > 500.0:
        score -= 2.0
    if float(max_red_grasper_normal_force) > 1000.0:
        score -= 3.0
    if float(hold_preclose_contact_ratio) > 0.05:
        score -= 3.0
    return float(score)


def rank_trials(trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        trials,
        key=lambda trial: (
            -float(trial.get("score", 0.0)),
            -float(trial.get("hold_closed_contact_ratio", 0.0)),
            float(trial.get("max_red_grasper_normal_force", float("inf"))),
            float(trial.get("min_tip_red_distance", float("inf"))),
        ),
    )


def select_best_clean_persistent_trial(trials: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        trial
        for trial in trials
        if bool(trial.get("clean_preclose"))
        and bool(trial.get("close_added_contact"))
        and bool(trial.get("persistent_post_close"))
        and bool(trial.get("force_not_extreme"))
    ]
    if not candidates:
        return None
    return rank_trials(candidates)[0]


def make_judgment(
    *,
    best_clean_persistent_trial: dict[str, Any] | None,
    any_clean_preclose: bool,
    any_close_added_contact: bool,
    any_persistent_post_close: bool,
    any_clean_persistent_contact: bool,
) -> str:
    if best_clean_persistent_trial is not None:
        return "[OK] found clean preclose with persistent post-close contact"
    if any_clean_preclose and any_close_added_contact:
        return "[WARN] found clean close contact but persistence or force needs tuning"
    if any_persistent_post_close and not any_clean_persistent_contact:
        return "[WARN] contact persists but preclose is not clean"
    return "[FAIL] no useful tuned pick parameters found"


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


def _record_step(
    *,
    phase: str,
    step_index: int,
    mujoco: Any,
    model: Any,
    data: Any,
    initial_red_position: Sequence[float],
    close_start_red_position: Sequence[float] | None,
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
        "red_displacement_from_close_start": 0.0
        if close_start_red_position is None
        else float(_distance(close_start_red_position, red_position)),
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


def summarize_phase(phase_name: str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    phase_records = [record for record in records if record["phase"] == phase_name]
    if not phase_records:
        return {
            "phase": phase_name,
            "contact_ratio": 0.0,
            "contact_step_count": 0,
            "min_tip_red_distance": None,
            "final_tip_red_distance": None,
            "max_red_grasper_normal_force": 0.0,
            "mean_red_grasper_normal_force": 0.0,
            "red_displacement_delta": 0.0,
        }
    contact_records = [record for record in phase_records if bool(record["has_red_grasper_contact"])]
    contact_forces = [float(record["red_grasper_normal_force"]) for record in contact_records]
    return {
        "phase": phase_name,
        "contact_ratio": float(
            sum(1 for record in phase_records if bool(record["has_red_grasper_contact"])) / len(phase_records)
        ),
        "contact_step_count": int(len(contact_records)),
        "min_tip_red_distance": float(min(record["tip_red_distance"] for record in phase_records)),
        "final_tip_red_distance": float(phase_records[-1]["tip_red_distance"]),
        "max_red_grasper_normal_force": float(max(contact_forces)) if contact_forces else 0.0,
        "mean_red_grasper_normal_force": float(sum(contact_forces) / len(contact_forces))
        if contact_forces
        else 0.0,
        "red_displacement_delta": float(phase_records[-1]["red_displacement"] - phase_records[0]["red_displacement"]),
    }


def _run_trial(
    trial_config: dict[str, float],
    *,
    model: Any,
    data: Any,
    mujoco: Any,
    settle_steps: int,
    approach_ramp_steps: int,
    hold_preclose_steps: int,
    close_ramp_steps: int,
    hold_closed_steps: int,
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
    if initial_red_position is None:
        raise RuntimeError("Missing red_object body pose.")

    preclose_section_angles = scale_section_angles(FULL_SECTION_ANGLES, scale=float(trial_config["preclose_scale"]))
    final_close_section_angles = scale_section_angles(
        FULL_SECTION_ANGLES,
        scale=float(trial_config["final_close_scale"]),
    )

    records: list[dict[str, Any]] = []
    close_start_red_position: list[float] | None = None

    for step_index in range(approach_ramp_steps):
        action = {
            "section_angles": _interpolate_section_angles(
                [0.0, 1.5708, 0.0, 1.5708, 0.0, 1.5708],
                preclose_section_angles,
                step_index=step_index,
                total_steps=approach_ramp_steps,
            ),
            "grip_command": 0.0,
            "grasper_rotation": _lerp(0.0, float(trial_config["approach_rotation"]), step_index, approach_ramp_steps),
        }
        _apply_feagine_action(robot, action)
        mujoco.mj_step(model, data)
        records.append(
            _record_step(
                phase="approach_ramp",
                step_index=step_index,
                mujoco=mujoco,
                model=model,
                data=data,
                initial_red_position=initial_red_position,
                close_start_red_position=close_start_red_position,
                grip_command=float(action["grip_command"]),
                section_angles=action["section_angles"],
                grasper_rotation=float(action["grasper_rotation"]),
            )
        )

    for step_index in range(hold_preclose_steps):
        action = {
            "section_angles": [float(value) for value in preclose_section_angles],
            "grip_command": 0.0,
            "grasper_rotation": float(trial_config["approach_rotation"]),
        }
        _apply_feagine_action(robot, action)
        mujoco.mj_step(model, data)
        records.append(
            _record_step(
                phase="hold_preclose",
                step_index=step_index,
                mujoco=mujoco,
                model=model,
                data=data,
                initial_red_position=initial_red_position,
                close_start_red_position=close_start_red_position,
                grip_command=float(action["grip_command"]),
                section_angles=action["section_angles"],
                grasper_rotation=float(action["grasper_rotation"]),
            )
        )

    close_start_red_position = _body_position(mujoco, model, data, "red_object")
    if close_start_red_position is None:
        raise RuntimeError("Missing red_object body pose at close start.")

    for step_index in range(close_ramp_steps):
        action = {
            "section_angles": _interpolate_section_angles(
                preclose_section_angles,
                final_close_section_angles,
                step_index=step_index,
                total_steps=close_ramp_steps,
            ),
            "grip_command": _lerp(0.0, 1.0, step_index, close_ramp_steps),
            "grasper_rotation": _lerp(
                float(trial_config["approach_rotation"]),
                float(trial_config["close_rotation"]),
                step_index,
                close_ramp_steps,
            ),
        }
        _apply_feagine_action(robot, action)
        mujoco.mj_step(model, data)
        records.append(
            _record_step(
                phase="close_ramp",
                step_index=step_index,
                mujoco=mujoco,
                model=model,
                data=data,
                initial_red_position=initial_red_position,
                close_start_red_position=close_start_red_position,
                grip_command=float(action["grip_command"]),
                section_angles=action["section_angles"],
                grasper_rotation=float(action["grasper_rotation"]),
            )
        )

    for step_index in range(hold_closed_steps):
        action = {
            "section_angles": [float(value) for value in final_close_section_angles],
            "grip_command": 1.0,
            "grasper_rotation": float(trial_config["close_rotation"]),
        }
        _apply_feagine_action(robot, action)
        mujoco.mj_step(model, data)
        records.append(
            _record_step(
                phase="hold_closed",
                step_index=step_index,
                mujoco=mujoco,
                model=model,
                data=data,
                initial_red_position=initial_red_position,
                close_start_red_position=close_start_red_position,
                grip_command=float(action["grip_command"]),
                section_angles=action["section_angles"],
                grasper_rotation=float(action["grasper_rotation"]),
            )
        )

    phase_summaries = {
        phase_name: summarize_phase(phase_name, records)
        for phase_name in ("approach_ramp", "hold_preclose", "close_ramp", "hold_closed")
    }
    max_red_grasper_normal_force = max(float(record["red_grasper_normal_force"]) for record in records)
    mean_red_grasper_normal_force = (
        sum(float(record["red_grasper_normal_force"]) for record in records if bool(record["has_red_grasper_contact"]))
        / max(1, sum(1 for record in records if bool(record["has_red_grasper_contact"])))
    )
    red_motion_after_close = max(float(record["red_displacement_from_close_start"]) for record in records)
    min_tip_red_distance = min(float(record["tip_red_distance"]) for record in records)

    flags = compute_trial_flags(
        hold_preclose_contact_ratio=float(phase_summaries["hold_preclose"]["contact_ratio"]),
        close_ramp_contact_ratio=float(phase_summaries["close_ramp"]["contact_ratio"]),
        hold_closed_contact_ratio=float(phase_summaries["hold_closed"]["contact_ratio"]),
        red_motion_after_close=float(red_motion_after_close),
        max_red_grasper_normal_force=float(max_red_grasper_normal_force),
    )
    score = compute_trial_score(
        clean_preclose=bool(flags["clean_preclose"]),
        close_added_contact=bool(flags["close_added_contact"]),
        persistent_post_close=bool(flags["persistent_post_close"]),
        red_object_moved_after_close=bool(flags["red_object_moved_after_close"]),
        force_not_extreme=bool(flags["force_not_extreme"]),
        min_tip_red_distance=float(min_tip_red_distance),
        max_red_grasper_normal_force=float(max_red_grasper_normal_force),
        hold_preclose_contact_ratio=float(phase_summaries["hold_preclose"]["contact_ratio"]),
    )

    return {
        "name": (
            f"scale{int(round(float(trial_config['preclose_scale']) * 100)):03d}_"
            f"arot{int(round(float(trial_config['approach_rotation']) * 100)):04d}_"
            f"crot{int(round(float(trial_config['close_rotation']) * 100)):04d}_"
            f"adv{int(round(float(trial_config['close_advance_scale']) * 100)):03d}"
        ),
        "preclose_scale": float(trial_config["preclose_scale"]),
        "approach_rotation": float(trial_config["approach_rotation"]),
        "close_rotation": float(trial_config["close_rotation"]),
        "close_advance_scale": float(trial_config["close_advance_scale"]),
        "final_close_scale": float(trial_config["final_close_scale"]),
        "clean_preclose": bool(flags["clean_preclose"]),
        "close_added_contact": bool(flags["close_added_contact"]),
        "persistent_post_close": bool(flags["persistent_post_close"]),
        "red_object_moved_after_close": bool(flags["red_object_moved_after_close"]),
        "force_not_extreme": bool(flags["force_not_extreme"]),
        "hold_closed_contact_ratio": float(phase_summaries["hold_closed"]["contact_ratio"]),
        "post_close_contact_ratio": float(
            (
                float(phase_summaries["close_ramp"]["contact_step_count"])
                + float(phase_summaries["hold_closed"]["contact_step_count"])
            )
            / float(close_ramp_steps + hold_closed_steps)
        ),
        "red_motion_after_close": float(red_motion_after_close),
        "max_red_grasper_normal_force": float(max_red_grasper_normal_force),
        "mean_red_grasper_normal_force": float(mean_red_grasper_normal_force),
        "min_tip_red_distance": float(min_tip_red_distance),
        "contact_pairs_seen": sorted({pair for record in records for pair in record["contact_pairs"]}),
        "phase_summaries": phase_summaries,
        "score": float(score),
    }


def tune_pick_parameters(
    scene_path: str | Path,
    *,
    settle_steps: int,
    approach_ramp_steps: int,
    hold_preclose_steps: int,
    close_ramp_steps: int,
    hold_closed_steps: int,
    max_trials: int,
) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model, data = _load_tabletop_scene(str(resolved_scene))

    try:
        _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc

    trial_grid = build_trial_grid(max_trials=max_trials)
    trials: list[dict[str, Any]] = []

    for index, trial_config in enumerate(trial_grid):
        trial = _run_trial(
            trial_config,
            model=model,
            data=data,
            mujoco=mujoco,
            settle_steps=settle_steps,
            approach_ramp_steps=approach_ramp_steps,
            hold_preclose_steps=hold_preclose_steps,
            close_ramp_steps=close_ramp_steps,
            hold_closed_steps=hold_closed_steps,
        )
        trials.append(trial)
        print(
            f"[TRIAL {index:03d}/{len(trial_grid):03d}] "
            f"scale={trial['preclose_scale']} "
            f"arot={trial['approach_rotation']} "
            f"crot={trial['close_rotation']} "
            f"adv={trial['close_advance_scale']} "
            f"clean={trial['clean_preclose']} "
            f"close_contact={trial['close_added_contact']} "
            f"hold_ratio={trial['hold_closed_contact_ratio']} "
            f"max_force={trial['max_red_grasper_normal_force']} "
            f"score={trial['score']}"
        )

    ranked_trials = rank_trials(trials)
    best_trial_by_score = ranked_trials[0] if ranked_trials else None
    best_clean_persistent_trial = select_best_clean_persistent_trial(ranked_trials)
    any_clean_preclose = any(bool(trial["clean_preclose"]) for trial in ranked_trials)
    any_close_added_contact = any(bool(trial["close_added_contact"]) for trial in ranked_trials)
    any_clean_persistent_contact = any(
        bool(trial["clean_preclose"]) and bool(trial["persistent_post_close"]) for trial in ranked_trials
    )
    any_force_not_extreme = any(bool(trial["force_not_extreme"]) for trial in ranked_trials)
    any_persistent_post_close = any(bool(trial["persistent_post_close"]) for trial in ranked_trials)
    judgment = make_judgment(
        best_clean_persistent_trial=best_clean_persistent_trial,
        any_clean_preclose=any_clean_preclose,
        any_close_added_contact=any_close_added_contact,
        any_persistent_post_close=any_persistent_post_close,
        any_clean_persistent_contact=any_clean_persistent_contact,
    )

    return {
        "scene": str(resolved_scene),
        "full_section_angles": [float(value) for value in FULL_SECTION_ANGLES],
        "trials": ranked_trials,
        "summary": {
            "num_trials": int(len(ranked_trials)),
            "best_trial_by_score": best_trial_by_score,
            "best_clean_persistent_trial": best_clean_persistent_trial,
            "top_trials": ranked_trials[:10],
            "flags": {
                "any_clean_preclose": bool(any_clean_preclose),
                "any_close_added_contact": bool(any_close_added_contact),
                "any_clean_persistent_contact": bool(any_clean_persistent_contact),
                "any_force_not_extreme": bool(any_force_not_extreme),
            },
        },
        "judgment": judgment,
        "notes": [
            "This is parameter tuning for clean-preclose and persistent post-close contact.",
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
        report = tune_pick_parameters(
            scene_path,
            settle_steps=args.settle_steps,
            approach_ramp_steps=args.approach_ramp_steps,
            hold_preclose_steps=args.hold_preclose_steps,
            close_ramp_steps=args.close_ramp_steps,
            hold_closed_steps=args.hold_closed_steps,
            max_trials=args.max_trials,
        )
    except ValueError as exc:
        if "unsupported action keys" in str(exc):
            print(f"[FAIL] {exc}")
            return 4
        print(f"[FAIL] parameter tuning failed: {exc}")
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
        print(f"[FAIL] parameter tuning failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] parameter tuning failed: {exc}")
        return 1

    summary = report["summary"]
    print(
        "[SUMMARY] best_trial_by_score="
        f"{None if summary['best_trial_by_score'] is None else summary['best_trial_by_score']['name']}"
    )
    print(
        "[SUMMARY] best_clean_persistent_trial="
        f"{None if summary['best_clean_persistent_trial'] is None else summary['best_clean_persistent_trial']['name']}"
    )
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
