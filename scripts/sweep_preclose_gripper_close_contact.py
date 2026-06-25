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
PRECLOSE_SCALES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CLOSE_ADVANCE_SCALES = [0.00, 0.05, 0.10, 0.15, 0.20]
GRASPER_ROTATIONS = [0.0, -0.7854, 0.7854, -1.5708, 1.5708]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep preclose scale, grasper rotation, and close micro-advance for clean close contact."
    )
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--approach-ramp-steps", type=int, default=120)
    parser.add_argument("--hold-preclose-steps", type=int, default=160)
    parser.add_argument("--close-ramp-steps", type=int, default=120)
    parser.add_argument("--hold-closed-steps", type=int, default=160)
    parser.add_argument("--max-trials", type=int, default=150)
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
    close_advance_scales: Sequence[float] = CLOSE_ADVANCE_SCALES,
    grasper_rotations: Sequence[float] = GRASPER_ROTATIONS,
    max_trials: int = 150,
) -> list[dict[str, float]]:
    grid: list[dict[str, float]] = []
    for preclose_scale in preclose_scales:
        for close_advance_scale in close_advance_scales:
            for grasper_rotation in grasper_rotations:
                final_close_scale = min(1.0, float(preclose_scale) + float(close_advance_scale))
                grid.append(
                    {
                        "preclose_scale": float(preclose_scale),
                        "close_advance_scale": float(close_advance_scale),
                        "final_close_scale": float(final_close_scale),
                        "grasper_rotation": float(grasper_rotation),
                    }
                )
                if len(grid) >= int(max_trials):
                    return grid
    return grid


def evaluate_trial_outcome(
    *,
    preclose_had_contact: bool,
    red_displacement_before_close: float,
    close_had_contact: bool,
    hold_closed_had_contact: bool,
    red_displacement_after_close: float,
    red_displacement_before_close_baseline: float,
) -> dict[str, Any]:
    clean_preclose = (not bool(preclose_had_contact)) and float(red_displacement_before_close) < 0.005
    close_added_contact = bool(clean_preclose and (close_had_contact or hold_closed_had_contact))
    close_added_red_displacement = float(red_displacement_after_close) - float(red_displacement_before_close_baseline)
    close_added_motion = bool(close_added_red_displacement > 0.003)
    return {
        "clean_preclose": bool(clean_preclose),
        "close_added_contact": bool(close_added_contact),
        "close_added_motion": bool(close_added_motion),
        "close_added_red_displacement": float(close_added_red_displacement),
    }


def rank_trials(trials: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        trials,
        key=lambda item: (
            0 if bool(item.get("close_added_contact")) else 1,
            0 if not bool(item.get("preclose_had_contact")) else 1,
            float(item.get("preclose_final_distance", float("inf"))),
            -float(item.get("close_added_red_displacement", 0.0)),
        ),
    )


def classify_contacts(contacts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    has_red_grasper_contact = False
    has_red_pedestal_contact = False
    max_red_grasper_normal_force = 0.0
    contact_details: list[dict[str, Any]] = []

    for contact in contacts:
        geom1 = str(contact.get("geom1", ""))
        geom2 = str(contact.get("geom2", ""))
        body1 = str(contact.get("body1", ""))
        body2 = str(contact.get("body2", ""))
        distance = float(contact.get("distance", 0.0) or 0.0)
        normal_force = abs(float(contact.get("normal_force", 0.0) or 0.0))
        labels = [geom1, geom2, body1, body2]
        has_red = any("red_object" in label.lower() for label in labels)
        has_pedestal = any("red_pedestal" in label.lower() for label in labels)
        has_grasper = any(
            needle in body.lower()
            for body in (body1, body2)
            for needle in ("feagine_grasper", "grasper", "finger")
        )

        has_red_pedestal_contact = has_red_pedestal_contact or (has_red and has_pedestal)
        if has_red and has_grasper:
            has_red_grasper_contact = True
            max_red_grasper_normal_force = max(max_red_grasper_normal_force, normal_force)
            contact_details.append(
                {
                    "geom_pair": [geom1, geom2],
                    "body_pair": [body1, body2],
                    "distance": float(distance),
                    "normal_force": float(normal_force),
                }
            )

    return {
        "has_red_grasper_contact": bool(has_red_grasper_contact),
        "has_red_pedestal_contact": bool(has_red_pedestal_contact),
        "max_red_grasper_normal_force": float(max_red_grasper_normal_force),
        "contact_details": contact_details,
    }


def make_judgment(
    *,
    any_red_grasper_contact: bool,
    best_clean_close_trial: dict[str, Any] | None,
) -> str:
    if best_clean_close_trial and bool(best_clean_close_trial.get("close_added_contact")):
        if float(best_clean_close_trial.get("close_advance_scale", 0.0)) == 0.0:
            return "[OK] clean close produced red-grasper contact"
        return "[WARN] close contact required section micro-advance"
    if any_red_grasper_contact:
        return "[WARN] only approach/direct contact observed; close alone is weak"
    return "[FAIL] no useful preclose-close contact found"


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
        "tip_red_distance": float(_distance(tip_position, red_position)),
        "red_displacement": float(_distance(initial_red_position, red_position)),
        "contact_pairs": sorted(
            {" <-> ".join(sorted([str(contact["geom1"]), str(contact["geom2"])])) for contact in contacts}
        ),
        "contact_details": flags["contact_details"],
        "has_red_grasper_contact": bool(flags["has_red_grasper_contact"]),
        "max_red_grasper_normal_force": float(flags["max_red_grasper_normal_force"]),
        "grip_command": float(grip_command),
        "section_angles": [float(value) for value in section_angles],
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


def _interpolate_section_angles(target_section_angles: Sequence[float], *, step_index: int, total_steps: int) -> list[float]:
    if total_steps <= 0:
        return [float(value) for value in target_section_angles]
    alpha = float(step_index + 1) / float(total_steps)
    return [float(alpha * float(value)) for value in target_section_angles]


def _interpolate_between(start_angles: Sequence[float], end_angles: Sequence[float], *, step_index: int, total_steps: int) -> list[float]:
    if total_steps <= 0:
        return [float(value) for value in end_angles]
    alpha = float(step_index + 1) / float(total_steps)
    return [
        float((1.0 - alpha) * float(start_value) + alpha * float(end_value))
        for start_value, end_value in zip(start_angles, end_angles)
    ]


def _interpolate_grip_command(*, step_index: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    return float((step_index + 1) / float(total_steps))


def _phase_summary(records: Sequence[dict[str, Any]], phase_name: str) -> dict[str, Any]:
    phase_records = [record for record in records if record["phase"] == phase_name]
    if not phase_records:
        return {
            "min_tip_red_distance": None,
            "final_tip_red_distance": None,
            "max_red_displacement": 0.0,
            "has_red_grasper_contact": False,
            "max_red_grasper_normal_force": 0.0,
            "contact_pairs_seen": [],
        }
    return {
        "min_tip_red_distance": float(min(record["tip_red_distance"] for record in phase_records)),
        "final_tip_red_distance": float(phase_records[-1]["tip_red_distance"]),
        "max_red_displacement": float(max(record["red_displacement"] for record in phase_records)),
        "has_red_grasper_contact": any(bool(record["has_red_grasper_contact"]) for record in phase_records),
        "max_red_grasper_normal_force": float(
            max(record["max_red_grasper_normal_force"] for record in phase_records)
        ),
        "contact_pairs_seen": sorted({pair for record in phase_records for pair in record["contact_pairs"]}),
    }


def _run_trial(
    *,
    model: Any,
    data: Any,
    mujoco: Any,
    settle_steps: int,
    trial_config: dict[str, float],
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
    final_close_section_angles = scale_section_angles(FULL_SECTION_ANGLES, scale=float(trial_config["final_close_scale"]))
    rotation = float(trial_config["grasper_rotation"])

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
            "grasper_rotation": rotation,
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
            "grasper_rotation": rotation,
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
            "section_angles": _interpolate_between(
                preclose_section_angles,
                final_close_section_angles,
                step_index=step_index,
                total_steps=total_steps,
            ),
            "grip_command": _interpolate_grip_command(step_index=step_index, total_steps=total_steps),
            "grasper_rotation": rotation,
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
            "section_angles": [float(value) for value in final_close_section_angles],
            "grip_command": 1.0,
            "grasper_rotation": rotation,
        },
    )

    preclose_summary = _phase_summary(records, "hold_preclose")
    close_summary = _phase_summary(records, "close_ramp")
    hold_closed_summary = _phase_summary(records, "hold_closed")
    all_contact_pairs = sorted({pair for record in records for pair in record["contact_pairs"]})
    outcome = evaluate_trial_outcome(
        preclose_had_contact=bool(preclose_summary["has_red_grasper_contact"]),
        red_displacement_before_close=float(preclose_summary["max_red_displacement"]),
        close_had_contact=bool(close_summary["has_red_grasper_contact"]),
        hold_closed_had_contact=bool(hold_closed_summary["has_red_grasper_contact"]),
        red_displacement_after_close=float(
            max(close_summary["max_red_displacement"], hold_closed_summary["max_red_displacement"])
        ),
        red_displacement_before_close_baseline=float(preclose_summary["max_red_displacement"]),
    )

    return {
        "name": (
            f"scale{int(round(float(trial_config['preclose_scale']) * 100)):03d}_"
            f"adv{int(round(float(trial_config['close_advance_scale']) * 100)):03d}_"
            f"rot{int(round(float(trial_config['grasper_rotation']) * 100)):04d}"
        ),
        "preclose_scale": float(trial_config["preclose_scale"]),
        "close_advance_scale": float(trial_config["close_advance_scale"]),
        "final_close_scale": float(trial_config["final_close_scale"]),
        "grasper_rotation": float(trial_config["grasper_rotation"]),
        "preclose_final_distance": float(preclose_summary["final_tip_red_distance"]),
        "preclose_had_contact": bool(preclose_summary["has_red_grasper_contact"]),
        "close_had_contact": bool(close_summary["has_red_grasper_contact"]),
        "hold_closed_had_contact": bool(hold_closed_summary["has_red_grasper_contact"]),
        "close_added_contact": bool(outcome["close_added_contact"]),
        "red_displacement_before_close": float(preclose_summary["max_red_displacement"]),
        "red_displacement_after_close": float(
            max(close_summary["max_red_displacement"], hold_closed_summary["max_red_displacement"])
        ),
        "close_added_red_displacement": float(outcome["close_added_red_displacement"]),
        "close_added_motion": bool(outcome["close_added_motion"]),
        "max_red_grasper_normal_force": float(
            max(
                preclose_summary["max_red_grasper_normal_force"],
                close_summary["max_red_grasper_normal_force"],
                hold_closed_summary["max_red_grasper_normal_force"],
            )
        ),
        "min_tip_red_distance_overall": float(min(record["tip_red_distance"] for record in records)),
        "contact_pairs_seen": all_contact_pairs,
    }


def sweep_preclose_close_trials(
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

    trials: list[dict[str, Any]] = []
    trial_grid = build_trial_grid(max_trials=max_trials)
    for index, trial_config in enumerate(trial_grid):
        summary = _run_trial(
            model=model,
            data=data,
            mujoco=mujoco,
            settle_steps=settle_steps,
            trial_config=trial_config,
            approach_ramp_steps=approach_ramp_steps,
            hold_preclose_steps=hold_preclose_steps,
            close_ramp_steps=close_ramp_steps,
            hold_closed_steps=hold_closed_steps,
        )
        trials.append(summary)
        print(
            f"[TRIAL {index:03d}/{len(trial_grid):03d}] "
            f"scale={summary['preclose_scale']} "
            f"adv={summary['close_advance_scale']} "
            f"rot={summary['grasper_rotation']} "
            f"pre_dist={summary['preclose_final_distance']} "
            f"pre_contact={summary['preclose_had_contact']} "
            f"close_contact={summary['close_had_contact'] or summary['hold_closed_had_contact']} "
            f"close_disp={summary['close_added_red_displacement']}"
        )

    ranked_trials = rank_trials(trials)
    best_clean_close_trial = next((trial for trial in ranked_trials if bool(trial["close_added_contact"])), None)
    best_near_preclose_trial = next(
        (trial for trial in ranked_trials if not bool(trial["preclose_had_contact"])),
        None,
    )
    any_clean_close_added_contact = any(bool(trial["close_added_contact"]) for trial in trials)
    any_close_added_motion = any(bool(trial["close_added_motion"]) for trial in trials)
    any_red_grasper_contact = any(
        bool(trial["preclose_had_contact"]) or bool(trial["close_had_contact"]) or bool(trial["hold_closed_had_contact"])
        for trial in trials
    )
    judgment = make_judgment(
        any_red_grasper_contact=any_red_grasper_contact,
        best_clean_close_trial=best_clean_close_trial,
    )

    return {
        "scene": str(resolved_scene),
        "full_section_angles": [float(value) for value in FULL_SECTION_ANGLES],
        "trials": trials,
        "summary": {
            "num_trials": int(len(trials)),
            "best_clean_close_trial": best_clean_close_trial,
            "best_near_preclose_trial": best_near_preclose_trial,
            "top_trials": ranked_trials[:10],
            "flags": {
                "any_clean_close_added_contact": bool(any_clean_close_added_contact),
                "any_close_added_motion": bool(any_close_added_motion),
                "any_red_grasper_contact": bool(any_red_grasper_contact),
            },
        },
        "judgment": judgment,
        "notes": [
            "This script sweeps clean preclose and close/micro-advance contact.",
            "It does not claim grasp success or object lifting.",
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
        report = sweep_preclose_close_trials(
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
        print(f"[FAIL] sweep failed: {exc}")
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
        print(f"[FAIL] sweep failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] sweep failed: {exc}")
        return 1

    summary = report["summary"]
    best_clean_close_trial = summary["best_clean_close_trial"]
    print(
        "[SUMMARY] best_clean_close_trial="
        f"{None if best_clean_close_trial is None else best_clean_close_trial['name']}"
    )
    print(f"[SUMMARY] any_clean_close_added_contact={summary['flags']['any_clean_close_added_contact']}")
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
