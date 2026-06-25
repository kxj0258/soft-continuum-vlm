from __future__ import annotations

import argparse
import math
import sys
import time
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
ROTATION_SETTLE_TARGET = -1.1781
ZERO_SECTION_ANGLES = [0.0, 1.5708, 0.0, 1.5708, 0.0, 1.5708]


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
            "scale_end": BASE_SCALE,
            "grip_start": 0.0,
            "grip_end": 1.0,
            "rotation_start": APPROACH_ROTATION,
            "rotation_end": CLOSE_ROTATION,
        },
        {
            "name": "hold_closed_initial",
            "steps": 240,
            "scale_start": BASE_SCALE,
            "scale_end": BASE_SCALE,
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
                    "scale_start": BASE_SCALE,
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
                    "scale_start": BASE_SCALE,
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
                    "rotation_end": ROTATION_SETTLE_TARGET,
                },
                {
                    "name": "rotation_hold",
                    "steps": 360,
                    "scale_start": BASE_SCALE,
                    "scale_end": BASE_SCALE,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": ROTATION_SETTLE_TARGET,
                    "rotation_end": ROTATION_SETTLE_TARGET,
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
                    "scale_start": BASE_SCALE,
                    "scale_end": 0.68,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": CLOSE_ROTATION,
                    "rotation_end": ROTATION_SETTLE_TARGET,
                },
                {
                    "name": "lift_like_hold",
                    "steps": 360,
                    "scale_start": 0.68,
                    "scale_end": 0.68,
                    "grip_start": 1.0,
                    "grip_end": 1.0,
                    "rotation_start": ROTATION_SETTLE_TARGET,
                    "rotation_end": ROTATION_SETTLE_TARGET,
                },
            ],
        },
    ]


def _parse_args() -> argparse.Namespace:
    trial_names = [schedule["name"] for schedule in build_trial_schedules()]
    parser = argparse.ArgumentParser(
        description="View the reachable red-object retract/lift-like diagnostic in MuJoCo."
    )
    parser.add_argument(
        "--scene",
        default="outputs/scenes/feagine_tabletop_scene_reachable.xml",
        help="Path to the MuJoCo scene XML.",
    )
    parser.add_argument(
        "--trial",
        choices=trial_names,
        default="gentle_retract_scale_down",
    )
    parser.add_argument("--realtime-factor", type=float, default=0.5)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 plays the full phase schedule; a positive value limits playback steps.",
    )
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


def scale_section_angles(section_angles: Sequence[float], scale: float) -> list[float]:
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


def _contact_snapshot(model: Any, data: Any) -> list[dict[str, str]]:
    contacts: list[dict[str, str]] = []
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


def _classify_contacts(contacts: Sequence[dict[str, str]]) -> dict[str, bool]:
    has_red_grasper_contact = False
    has_red_pedestal_contact = False

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
        has_grasper = any(
            needle in label
            for label in lowered
            for needle in ("feagine_grasper", "grasper", "finger")
        )
        has_red_grasper_contact = has_red_grasper_contact or (has_red and has_grasper)
        has_red_pedestal_contact = has_red_pedestal_contact or (has_red and has_pedestal)

    return {
        "has_red_grasper_contact": bool(has_red_grasper_contact),
        "has_red_pedestal_contact": bool(has_red_pedestal_contact),
    }


def _distance(position_a: Sequence[float], position_b: Sequence[float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray(position_a, dtype=np.float64) - np.asarray(position_b, dtype=np.float64)
        )
    )


def _viewer_is_running(viewer: Any) -> bool:
    is_running = getattr(viewer, "is_running", None)
    if callable(is_running):
        try:
            return bool(is_running())
        except Exception:
            return False
    return True


def _viewer_sync(viewer: Any) -> bool:
    sync = getattr(viewer, "sync", None)
    if not callable(sync):
        return True
    try:
        sync()
    except Exception:
        return False
    return True


def _close_viewer(viewer: Any) -> None:
    close = getattr(viewer, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _sleep(realtime_factor: float) -> None:
    if realtime_factor > 0.0:
        time.sleep(max(0.0, 0.03 / float(realtime_factor)))


def _selected_schedule(trial_name: str) -> dict[str, Any]:
    for schedule in build_trial_schedules():
        if schedule["name"] == trial_name:
            return schedule
    raise ValueError(f"unknown trial: {trial_name}")


def _make_action(phase: dict[str, Any], step_index: int) -> dict[str, Any]:
    steps = int(phase["steps"])
    start_section = (
        ZERO_SECTION_ANGLES
        if float(phase["scale_start"]) == 0.0
        else scale_section_angles(FULL_SECTION_ANGLES, float(phase["scale_start"]))
    )
    end_section = scale_section_angles(FULL_SECTION_ANGLES, float(phase["scale_end"]))
    return {
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


def _print_step(
    *,
    step_index: int,
    phase_name: str,
    mujoco: Any,
    model: Any,
    data: Any,
    grip_command: float,
) -> None:
    tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
    red_position = _body_position(mujoco, model, data, "red_object")
    contacts = _contact_snapshot(model, data)
    flags = _classify_contacts(contacts)
    distance = None if tip_position is None or red_position is None else _distance(tip_position, red_position)
    print(
        f"[STEP {step_index:04d}] "
        f"phase={phase_name} "
        f"tip={tip_position} "
        f"red={red_position} "
        f"dist={distance} "
        f"grip={float(grip_command)} "
        f"contact_red_grasper={flags['has_red_grasper_contact']} "
        f"contact_red_pedestal={flags['has_red_pedestal_contact']}"
    )


def run_viewer_demo(
    *,
    scene_path: str | Path,
    trial_name: str,
    realtime_factor: float,
    print_every: int,
    max_steps: int,
) -> int:
    import mujoco
    import mujoco.viewer

    model, data = _load_tabletop_scene(str(scene_path))
    try:
        robot = _create_feagine_wrapper(model, data)
    except Exception as exc:
        raise RuntimeError(f"wrapper creation failed: {exc}") from exc
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)

    schedule = _selected_schedule(trial_name)
    total_phase_steps = sum(int(phase["steps"]) for phase in schedule["phases"])
    playback_limit = total_phase_steps if int(max_steps) <= 0 else min(int(max_steps), total_phase_steps)
    print_interval = max(1, int(print_every))

    print("[INFO] Viewer demo started.")
    print("[INFO] Watch whether red_object is enveloped by fingers or pushed by arm/body.")
    print("[INFO] Watch whether red_object remains on pedestal or shows lift-like motion.")
    print("[INFO] This demo does not claim stable grasp or object lifting.")
    print(f"[INFO] trial={trial_name} max_steps={playback_limit} realtime_factor={float(realtime_factor)}")

    try:
        for _ in range(80):
            if not _viewer_is_running(viewer):
                return 0
            mujoco.mj_step(model, data)
            if not _viewer_sync(viewer):
                return 0
            _sleep(realtime_factor)

        global_step = 0
        for phase in schedule["phases"]:
            phase_name = str(phase["name"])
            for step_index in range(int(phase["steps"])):
                if global_step >= playback_limit:
                    print("[OK] reachable red pick viewer demo finished")
                    return 0
                if not _viewer_is_running(viewer):
                    print("[OK] reachable red pick viewer demo finished")
                    return 0

                action = _make_action(phase, step_index)
                try:
                    _apply_feagine_action(robot, action)
                except Exception as exc:
                    raise RuntimeError(f"action apply failed: {exc}") from exc
                mujoco.mj_step(model, data)
                if not _viewer_sync(viewer):
                    print("[OK] reachable red pick viewer demo finished")
                    return 0
                if global_step % print_interval == 0:
                    _print_step(
                        step_index=global_step,
                        phase_name=phase_name,
                        mujoco=mujoco,
                        model=model,
                        data=data,
                        grip_command=float(action["grip_command"]),
                    )
                global_step += 1
                _sleep(realtime_factor)
    except KeyboardInterrupt:
        pass
    finally:
        _close_viewer(viewer)

    print("[OK] reachable red pick viewer demo finished")
    return 0


def main() -> int:
    args = _parse_args()
    scene_path = Path(args.scene).expanduser().resolve()

    try:
        return run_viewer_demo(
            scene_path=scene_path,
            trial_name=str(args.trial),
            realtime_factor=float(args.realtime_factor),
            print_every=int(args.print_every),
            max_steps=int(args.max_steps),
        )
    except ValueError as exc:
        if "unsupported action keys" in str(exc):
            print(f"[FAIL] {exc}")
            return 3
        print(f"[FAIL] viewer demo failed: {exc}")
        return 1
    except RuntimeError as exc:
        message = str(exc)
        if "wrapper creation failed" in message:
            print("[FAIL] Failed to create Feagine wrapper for tabletop scene.")
            print(f"[DETAIL] {message.removeprefix('wrapper creation failed: ')}")
            return 2
        if "action apply failed" in message or "Missing Feagine wrapper" in message or "section_angles" in message:
            print("[FAIL] Could not apply Feagine action to tabletop scene.")
            print(f"[DETAIL] {message}")
            return 3
        print(f"[FAIL] viewer demo failed: {message}")
        return 1
    except Exception as exc:
        print(f"[FAIL] viewer demo failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
