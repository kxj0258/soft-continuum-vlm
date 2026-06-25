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
PRECLOSE_SCALES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether controlled close adds contact beyond direct local contact reproduction."
    )
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--direct-steps", type=int, default=160)
    parser.add_argument("--ramp-steps", type=int, default=160)
    parser.add_argument("--long-hold-steps", type=int, default=400)
    parser.add_argument("--preclose-ramp-steps", type=int, default=120)
    parser.add_argument("--preclose-hold-steps", type=int, default=200)
    parser.add_argument("--close-ramp-steps", type=int, default=120)
    parser.add_argument("--hold-closed-steps", type=int, default=200)
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
    contact_details: list[dict[str, Any]] = []

    for contact in contacts:
        geom1 = str(contact.get("geom1", ""))
        geom2 = str(contact.get("geom2", ""))
        body1 = str(contact.get("body1", ""))
        body2 = str(contact.get("body2", ""))
        distance = float(contact.get("distance", 0.0) or 0.0)
        normal_force = abs(float(contact.get("normal_force", 0.0) or 0.0))

        geom_pair = [geom1, geom2]
        body_pair = [body1, body2]
        labels = geom_pair + body_pair
        has_red = any(_name_matches(label, ("red_object",)) for label in labels)
        has_pedestal = any(_name_matches(label, ("red_pedestal",)) for label in labels)
        has_table = any(_name_matches(label, ("tabletop", "table")) for label in labels)
        has_grasper = any(_name_matches(label, ("feagine_grasper", "grasper", "finger")) for label in body_pair)

        has_red_pedestal_contact = has_red_pedestal_contact or (has_red and has_pedestal)
        has_red_table_contact = has_red_table_contact or (has_red and has_table)
        if has_red and has_grasper:
            has_red_grasper_contact = True
            max_red_grasper_normal_force = max(max_red_grasper_normal_force, normal_force)
            total_red_grasper_normal_force += normal_force
            contact_details.append(
                {
                    "geom_pair": geom_pair,
                    "body_pair": body_pair,
                    "distance": float(distance),
                    "normal_force": float(normal_force),
                }
            )

    return {
        "has_red_grasper_contact": bool(has_red_grasper_contact),
        "has_red_pedestal_contact": bool(has_red_pedestal_contact),
        "has_red_table_contact": bool(has_red_table_contact),
        "max_red_grasper_normal_force": float(max_red_grasper_normal_force),
        "total_red_grasper_normal_force": float(total_red_grasper_normal_force),
        "contact_details": contact_details,
    }


def select_preclose_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No preclose candidates available.")

    clean_candidates = [
        candidate
        for candidate in candidates
        if (not bool(candidate.get("has_red_grasper_contact")))
        and 0.05 <= float(candidate.get("final_tip_red_distance", float("inf"))) <= 0.15
        and float(candidate.get("max_red_displacement", float("inf"))) < 0.005
    ]
    if clean_candidates:
        best = min(
            clean_candidates,
            key=lambda item: (
                float(item["final_tip_red_distance"]),
                float(item["max_red_displacement"]),
                -float(item["scale"]),
            ),
        )
        return {**best, "clean_preclose_found": True}

    no_contact_candidates = [
        candidate for candidate in candidates if not bool(candidate.get("has_red_grasper_contact"))
    ]
    if no_contact_candidates:
        best = min(
            no_contact_candidates,
            key=lambda item: (
                float(item["final_tip_red_distance"]),
                float(item["max_red_displacement"]),
                -float(item["scale"]),
            ),
        )
        return {**best, "clean_preclose_found": False}

    best = min(
        candidates,
        key=lambda item: (
            float(item["final_tip_red_distance"]),
            float(item["max_red_displacement"]),
            -float(item["scale"]),
        ),
    )
    return {**best, "clean_preclose_found": False}


def build_comparison(
    *,
    direct_open_repro: dict[str, Any],
    direct_close_repro: dict[str, Any],
    preclose_scaled_close: dict[str, Any],
) -> dict[str, Any]:
    return {
        "direct_close_minus_open_red_displacement": float(direct_close_repro["max_red_displacement"])
        - float(direct_open_repro["max_red_displacement"]),
        "direct_close_minus_open_max_force": float(direct_close_repro["max_red_grasper_normal_force"])
        - float(direct_open_repro["max_red_grasper_normal_force"]),
        "preclose_close_added_contact": bool(preclose_scaled_close["close_added_contact"]),
        "preclose_close_added_red_displacement": float(preclose_scaled_close["close_added_red_displacement"]),
    }


def make_judgment(
    *,
    direct_open_repro: dict[str, Any],
    direct_close_repro: dict[str, Any],
    preclose_scaled_close: dict[str, Any],
) -> str:
    preclose_added_contact = (
        not bool(preclose_scaled_close.get("preclose_had_contact"))
        and (
            bool(preclose_scaled_close.get("has_red_grasper_contact_during_close"))
            or bool(preclose_scaled_close.get("has_red_grasper_contact_during_hold_closed"))
        )
    )
    if preclose_added_contact:
        return "[OK] controlled close added contact from a no-contact preclose state"

    direct_contact_reproduced = bool(direct_open_repro.get("has_red_grasper_contact")) or bool(
        direct_close_repro.get("has_red_grasper_contact")
    )
    if direct_contact_reproduced and bool(preclose_scaled_close.get("clean_preclose_found")):
        return "[WARN] contact reproduced, but close effect is ambiguous"
    if direct_contact_reproduced:
        return "[WARN] direct contact reproduced, but no clean preclose state found"
    return "[FAIL] could not reproduce CONTACT-3B contact"


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
                "distance": float(contact.dist),
                "normal_force": 0.0 if force6 is None else float(force6[0]),
            }
        )
    return contacts


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
) -> dict[str, Any]:
    tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    red_position = _body_position(mujoco, model, data, "red_object")
    if tip_position is None or red_position is None:
        raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")
    contacts = _contact_snapshot(mujoco, model, data)
    flags = classify_contacts(contacts)
    return {
        "phase": phase,
        "step_index": int(step_index),
        "tip_position": [float(value) for value in tip_position],
        "red_position": [float(value) for value in red_position],
        "tip_red_distance": float(_distance(tip_position, red_position)),
        "red_displacement": float(_distance(initial_red_position, red_position)),
        "grip_command": float(grip_command),
        "section_angles": [float(value) for value in section_angles],
        "contact_pairs": sorted(
            {" <-> ".join(sorted([str(contact["geom1"]), str(contact["geom2"])])) for contact in contacts}
        ),
        "contact_details": flags["contact_details"],
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
                step_index=step_index,
                mujoco=mujoco,
                model=model,
                data=data,
                initial_red_position=initial_red_position,
                grip_command=float(action["grip_command"]),
                section_angles=action["section_angles"],
            )
        )


def _summarize_records(
    *,
    name: str,
    initial_tip_position: Sequence[float],
    final_tip_position: Sequence[float],
    initial_red_position: Sequence[float],
    final_red_position: Sequence[float],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    min_distance_record = min(records, key=lambda item: float(item["tip_red_distance"]))
    min_tip_red_distance = float(min_distance_record["tip_red_distance"])
    final_tip_red_distance = float(records[-1]["tip_red_distance"]) if records else float("inf")
    max_red_displacement = max(float(record["red_displacement"]) for record in records) if records else 0.0
    max_red_grasper_normal_force = max(
        float(record["max_red_grasper_normal_force"]) for record in records
    ) if records else 0.0
    contact_pairs_seen = sorted({pair for record in records for pair in record.get("contact_pairs", [])})
    contact_details_seen: list[dict[str, Any]] = []
    seen_details: set[tuple[str, str, str, str]] = set()
    for record in records:
        for detail in record.get("contact_details", []):
            key = (
                str(detail["geom_pair"][0]),
                str(detail["geom_pair"][1]),
                str(detail["body_pair"][0]),
                str(detail["body_pair"][1]),
            )
            if key in seen_details:
                continue
            seen_details.add(key)
            contact_details_seen.append(detail)

    def phase_contact(phase_name: str) -> bool:
        return any(bool(record["has_red_grasper_contact"]) for record in records if record["phase"] == phase_name)

    def phase_max_displacement(phase_name: str) -> float:
        values = [float(record["red_displacement"]) for record in records if record["phase"] == phase_name]
        return float(max(values)) if values else 0.0

    return {
        "name": name,
        "initial_red_position": [float(value) for value in initial_red_position],
        "final_red_position": [float(value) for value in final_red_position],
        "initial_tip_position": [float(value) for value in initial_tip_position],
        "final_tip_position": [float(value) for value in final_tip_position],
        "min_tip_red_distance": float(min_tip_red_distance),
        "final_tip_red_distance": float(final_tip_red_distance),
        "max_red_displacement": float(max_red_displacement),
        "max_red_grasper_normal_force": float(max_red_grasper_normal_force),
        "has_red_grasper_contact": any(bool(record["has_red_grasper_contact"]) for record in records),
        "contact_pairs_seen": contact_pairs_seen,
        "contact_details_seen": contact_details_seen,
        "phase_stats": {
            "approach_ramp": {
                "has_red_grasper_contact": phase_contact("approach_ramp"),
                "max_red_displacement": phase_max_displacement("approach_ramp"),
            },
            "hold_open": {
                "has_red_grasper_contact": phase_contact("hold_open"),
                "max_red_displacement": phase_max_displacement("hold_open"),
            },
            "close_ramp": {
                "has_red_grasper_contact": phase_contact("close_ramp"),
                "max_red_displacement": phase_max_displacement("close_ramp"),
            },
            "hold_closed": {
                "has_red_grasper_contact": phase_contact("hold_closed"),
                "max_red_displacement": phase_max_displacement("hold_closed"),
            },
            "hold": {
                "has_red_grasper_contact": phase_contact("hold"),
                "max_red_displacement": phase_max_displacement("hold"),
            },
        },
        "records": list(records),
    }


def _run_trial(
    *,
    name: str,
    model: Any,
    data: Any,
    mujoco: Any,
    settle_steps: int,
    phases: Sequence[tuple[str, int, Any]],
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
    for phase_name, steps, action_factory in phases:
        _run_phase(
            phase_name=phase_name,
            steps=steps,
            robot=robot,
            mujoco=mujoco,
            model=model,
            data=data,
            initial_red_position=initial_red_position,
            action_factory=action_factory,
            records=records,
        )

    final_tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    final_red_position = _body_position(mujoco, model, data, "red_object")
    if final_tip_position is None or final_red_position is None:
        raise RuntimeError("Missing feagine_grasper_tip or red_object body pose.")

    return _summarize_records(
        name=name,
        initial_tip_position=initial_tip_position,
        final_tip_position=final_tip_position,
        initial_red_position=initial_red_position,
        final_red_position=final_red_position,
        records=records,
    )


def _open_target_action() -> dict[str, Any]:
    return {
        "section_angles": [float(value) for value in TARGET_SECTION_ANGLES],
        "grip_command": 0.0,
        "grasper_rotation": float(TARGET_GRASPER_ROTATION),
    }


def _close_target_action() -> dict[str, Any]:
    return {
        "section_angles": [float(value) for value in TARGET_SECTION_ANGLES],
        "grip_command": 1.0,
        "grasper_rotation": float(TARGET_GRASPER_ROTATION),
    }


def _constant_action_factory(action: dict[str, Any]):
    payload = {
        "section_angles": [float(value) for value in action["section_angles"]],
        "grip_command": float(action["grip_command"]),
        "grasper_rotation": float(action["grasper_rotation"]),
    }
    return lambda step_index, total_steps: dict(payload)


def _scaled_open_action(scale: float) -> dict[str, Any]:
    return {
        "section_angles": scale_section_angles(TARGET_SECTION_ANGLES, scale=float(scale)),
        "grip_command": 0.0,
        "grasper_rotation": float(TARGET_GRASPER_ROTATION),
    }


def _summarize_preclose_search_trial(scale: float, trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": float(scale),
        "min_tip_red_distance": float(trial["min_tip_red_distance"]),
        "final_tip_red_distance": float(trial["final_tip_red_distance"]),
        "has_red_grasper_contact": bool(trial["has_red_grasper_contact"]),
        "max_red_displacement": float(trial["max_red_displacement"]),
    }


def run_controlled_close_local_diagnostic(
    scene_path: str | Path,
    *,
    settle_steps: int,
    direct_steps: int,
    ramp_steps: int,
    long_hold_steps: int,
    preclose_ramp_steps: int,
    preclose_hold_steps: int,
    close_ramp_steps: int,
    hold_closed_steps: int,
) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model, data = _load_tabletop_scene(str(resolved_scene))

    try:
        _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc

    open_target_action = _open_target_action()
    close_target_action = _close_target_action()

    direct_open_repro = _run_trial(
        name="direct_open_repro",
        model=model,
        data=data,
        mujoco=mujoco,
        settle_steps=settle_steps,
        phases=[
            ("hold", direct_steps, _constant_action_factory(open_target_action)),
        ],
    )

    direct_close_repro = _run_trial(
        name="direct_close_repro",
        model=model,
        data=data,
        mujoco=mujoco,
        settle_steps=settle_steps,
        phases=[
            ("hold", direct_steps, _constant_action_factory(close_target_action)),
        ],
    )

    ramp_then_long_hold_open = _run_trial(
        name="ramp_then_long_hold_open",
        model=model,
        data=data,
        mujoco=mujoco,
        settle_steps=settle_steps,
        phases=[
            (
                "approach_ramp",
                ramp_steps,
                lambda step_index, total_steps: {
                    "section_angles": interpolate_section_angles(
                        TARGET_SECTION_ANGLES,
                        step_index=step_index,
                        total_steps=total_steps,
                    ),
                    "grip_command": 0.0,
                    "grasper_rotation": float(TARGET_GRASPER_ROTATION),
                },
            ),
            ("hold_open", long_hold_steps, _constant_action_factory(open_target_action)),
        ],
    )

    scaled_preclose_search: list[dict[str, Any]] = []
    for scale in PRECLOSE_SCALES:
        scaled_open_action = _scaled_open_action(scale)
        search_trial = _run_trial(
            name=f"scaled_preclose_{scale:.2f}",
            model=model,
            data=data,
            mujoco=mujoco,
            settle_steps=settle_steps,
            phases=[
                (
                    "approach_ramp",
                    preclose_ramp_steps,
                    lambda step_index, total_steps, action=scaled_open_action: {
                        "section_angles": interpolate_section_angles(
                            action["section_angles"],
                            step_index=step_index,
                            total_steps=total_steps,
                        ),
                        "grip_command": 0.0,
                        "grasper_rotation": float(TARGET_GRASPER_ROTATION),
                    },
                ),
                ("hold_open", preclose_hold_steps, _constant_action_factory(scaled_open_action)),
            ],
        )
        scaled_preclose_search.append(_summarize_preclose_search_trial(scale, search_trial))

    selected_preclose = select_preclose_candidate(scaled_preclose_search)
    selected_scale = float(selected_preclose["scale"])
    scaled_open_action = _scaled_open_action(selected_scale)
    preclose_scaled_close_trial = _run_trial(
        name="preclose_scaled_close",
        model=model,
        data=data,
        mujoco=mujoco,
        settle_steps=settle_steps,
        phases=[
            (
                "approach_ramp",
                preclose_ramp_steps,
                lambda step_index, total_steps, action=scaled_open_action: {
                    "section_angles": interpolate_section_angles(
                        action["section_angles"],
                        step_index=step_index,
                        total_steps=total_steps,
                    ),
                    "grip_command": 0.0,
                    "grasper_rotation": float(TARGET_GRASPER_ROTATION),
                },
            ),
            ("hold_open", preclose_hold_steps, _constant_action_factory(scaled_open_action)),
            (
                "close_ramp",
                close_ramp_steps,
                lambda step_index, total_steps, action=scaled_open_action: {
                    "section_angles": [float(value) for value in action["section_angles"]],
                    "grip_command": interpolate_grip_command(step_index=step_index, total_steps=total_steps),
                    "grasper_rotation": float(TARGET_GRASPER_ROTATION),
                },
            ),
            (
                "hold_closed",
                hold_closed_steps,
                lambda step_index, total_steps, action=scaled_open_action: {
                    "section_angles": [float(value) for value in action["section_angles"]],
                    "grip_command": 1.0,
                    "grasper_rotation": float(TARGET_GRASPER_ROTATION),
                },
            ),
        ],
    )

    preclose_scaled_close = {
        "name": "preclose_scaled_close",
        "selected_scale": selected_scale,
        "clean_preclose_found": bool(selected_preclose["clean_preclose_found"]),
        "preclose_final_distance": float(selected_preclose["final_tip_red_distance"]),
        "preclose_had_contact": bool(selected_preclose["has_red_grasper_contact"]),
        "close_added_contact": bool(
            (
                preclose_scaled_close_trial["phase_stats"]["close_ramp"]["has_red_grasper_contact"]
                or preclose_scaled_close_trial["phase_stats"]["hold_closed"]["has_red_grasper_contact"]
            )
            and not bool(selected_preclose["has_red_grasper_contact"])
        ),
        "close_added_red_displacement": float(preclose_scaled_close_trial["max_red_displacement"])
        - float(selected_preclose["max_red_displacement"]),
        "close_max_force": float(preclose_scaled_close_trial["max_red_grasper_normal_force"]),
        "has_red_grasper_contact_during_close": bool(
            preclose_scaled_close_trial["phase_stats"]["close_ramp"]["has_red_grasper_contact"]
        ),
        "has_red_grasper_contact_during_hold_closed": bool(
            preclose_scaled_close_trial["phase_stats"]["hold_closed"]["has_red_grasper_contact"]
        ),
        "trial_summary": preclose_scaled_close_trial,
    }

    comparison = build_comparison(
        direct_open_repro=direct_open_repro,
        direct_close_repro=direct_close_repro,
        preclose_scaled_close=preclose_scaled_close,
    )
    judgment = make_judgment(
        direct_open_repro=direct_open_repro,
        direct_close_repro=direct_close_repro,
        preclose_scaled_close=preclose_scaled_close,
    )

    notes = [
        "This script separates direct contact reproduction from controlled pre-close closing.",
        "It does not claim grasp success or object lifting.",
    ]
    if not direct_open_repro["has_red_grasper_contact"]:
        notes.append("[WARN] direct_open_repro did not reproduce CONTACT-3B; check action apply/reset differences.")

    return {
        "scene": str(resolved_scene),
        "target_action": {
            "open_target_action": open_target_action,
            "close_target_action": close_target_action,
        },
        "trials": {
            "direct_open_repro": {k: v for k, v in direct_open_repro.items() if k != "records"},
            "direct_close_repro": {k: v for k, v in direct_close_repro.items() if k != "records"},
            "ramp_then_long_hold_open": {k: v for k, v in ramp_then_long_hold_open.items() if k != "records"},
            "scaled_preclose_search": scaled_preclose_search,
            "preclose_scaled_close": {
                **{k: v for k, v in preclose_scaled_close.items() if k != "trial_summary"},
                "trial_summary": {k: v for k, v in preclose_scaled_close_trial.items() if k != "records"},
            },
        },
        "comparison": comparison,
        "judgment": judgment,
        "notes": notes,
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
        report = run_controlled_close_local_diagnostic(
            scene_path,
            settle_steps=args.settle_steps,
            direct_steps=args.direct_steps,
            ramp_steps=args.ramp_steps,
            long_hold_steps=args.long_hold_steps,
            preclose_ramp_steps=args.preclose_ramp_steps,
            preclose_hold_steps=args.preclose_hold_steps,
            close_ramp_steps=args.close_ramp_steps,
            hold_closed_steps=args.hold_closed_steps,
        )
    except ValueError as exc:
        if "unsupported action keys" in str(exc):
            print(f"[FAIL] {exc}")
            return 4
        print(f"[FAIL] controlled close local test failed: {exc}")
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if "wrapper creation failed" in message:
            print("[FAIL] Failed to create Feagine wrapper for tabletop scene.")
            print(f"[DETAIL] {message.removeprefix('wrapper creation failed: ')}")
            return 2
        if "unsupported action keys" in message:
            print(f"[FAIL] {message}")
            return 4
        if "Missing Feagine wrapper" in message or "section_angles" in message:
            print("[FAIL] Could not apply Feagine action to tabletop scene.")
            print(f"[DETAIL] {message}")
            return 3
        print(f"[FAIL] controlled close local test failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] controlled close local test failed: {exc}")
        return 1

    direct_open = report["trials"]["direct_open_repro"]
    direct_close = report["trials"]["direct_close_repro"]
    ramp_hold = report["trials"]["ramp_then_long_hold_open"]
    preclose = report["trials"]["preclose_scaled_close"]

    print(
        f"[TRIAL] direct_open_repro min_dist={direct_open['min_tip_red_distance']} "
        f"contact={direct_open['has_red_grasper_contact']} "
        f"red_disp={direct_open['max_red_displacement']}"
    )
    print(
        f"[TRIAL] direct_close_repro min_dist={direct_close['min_tip_red_distance']} "
        f"contact={direct_close['has_red_grasper_contact']} "
        f"red_disp={direct_close['max_red_displacement']}"
    )
    print(
        f"[TRIAL] ramp_then_long_hold_open min_dist={ramp_hold['min_tip_red_distance']} "
        f"contact_ramp={ramp_hold['phase_stats']['approach_ramp']['has_red_grasper_contact']} "
        f"contact_hold={ramp_hold['phase_stats']['hold_open']['has_red_grasper_contact']}"
    )
    for candidate in report["trials"]["scaled_preclose_search"]:
        print(
            f"[SEARCH] scale={candidate['scale']} "
            f"final_dist={candidate['final_tip_red_distance']} "
            f"contact={candidate['has_red_grasper_contact']} "
            f"red_disp={candidate['max_red_displacement']}"
        )
    print(f"[SELECT] preclose scale={preclose['selected_scale']}")
    print(
        f"[TRIAL] preclose_scaled_close preclose_dist={preclose['preclose_final_distance']} "
        f"close_contact={preclose['has_red_grasper_contact_during_close']} "
        f"hold_closed_contact={preclose['has_red_grasper_contact_during_hold_closed']}"
    )
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
