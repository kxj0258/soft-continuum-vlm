from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.ui.tk_control_panel import TkFeagineControlPanel


ALLOWED_ACTION_KEYS = {"section_angles", "grip_command", "grasper_rotation"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control the Feagine tabletop scene with a Tkinter control panel."
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="outputs/scenes/feagine_tabletop_scene.xml",
    )
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument(
        "--panel-title",
        type=str,
        default="Feagine Tabletop Panel Control",
    )
    return parser.parse_args()


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


def _as_float_sequence(value: Any, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{label} must be a numeric sequence.")
    return [float(item) for item in value]


def _apply_feagine_action(robot: Any, action: dict[str, Any]) -> None:
    unsupported = sorted(set(action) - ALLOWED_ACTION_KEYS)
    if unsupported:
        raise ValueError(f"unsupported action keys: {unsupported}")

    if "section_angles" in action:
        drive_section_angles = getattr(robot, "drive_section_angles", None)
        section_count = getattr(robot, "section_count", None)
        if not callable(drive_section_angles) or section_count is None:
            raise RuntimeError(
                "Missing Feagine wrapper section angle interface: "
                "drive_section_angles/section_count."
            )
        section_angles = _as_float_sequence(action["section_angles"], "section_angles")
        expected = int(section_count) * 2
        if len(section_angles) != expected:
            raise ValueError(
                f"section_angles must contain {expected} values, got {len(section_angles)}."
            )
        drive_section_angles(section_angles)

    if "grip_command" in action:
        set_grip_command = getattr(robot, "set_grip_command", None)
        if not callable(set_grip_command):
            raise RuntimeError(
                "Missing Feagine wrapper gripper interface: set_grip_command."
            )
        set_grip_command(float(action["grip_command"]))

    if "grasper_rotation" in action:
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
    position = data.xpos[body_id]
    return [float(position[index]) for index in range(3)]


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


def _is_paused(panel: TkFeagineControlPanel) -> bool:
    controller = getattr(panel, "controller", None)
    state = getattr(controller, "state", None)
    return bool(getattr(state, "paused", False))


def _sleep(realtime_factor: float) -> None:
    if realtime_factor > 0.0:
        time.sleep(max(0.0, 0.03 / float(realtime_factor)))


def main() -> int:
    args = _parse_args()
    model = None
    data = None
    robot = None
    viewer = None
    panel: TkFeagineControlPanel | None = None
    print_every = max(1, int(args.print_every))
    scene_path = Path(args.scene).expanduser().resolve()

    try:
        import mujoco
    except ImportError as exc:
        print(f"[FAIL] MuJoCo runtime import failed: {exc}")
        return 2

    try:
        model, data = _load_tabletop_scene(str(scene_path))
        robot = _create_feagine_wrapper(model, data)
        mujoco.mj_forward(model, data)
    except Exception as exc:
        print("[FAIL] Failed to create Feagine wrapper for tabletop scene.")
        print(f"[DETAIL] {exc}")
        return 2

    try:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        print("[FAIL] Failed to open MuJoCo viewer.")
        print(f"[DETAIL] {exc}")
        return 1

    try:
        panel = TkFeagineControlPanel(title=args.panel_title)
    except Exception as exc:
        print("[FAIL] Failed to open Tkinter control panel.")
        print(f"[DETAIL] {exc}")
        _close_viewer(viewer)
        return 3

    print("[INFO] Opened MuJoCo viewer and Tkinter control panel.")
    print("[INFO] Use the control panel sliders/buttons to command Feagine.")

    try:
        for step_index in range(int(args.max_steps)):
            if not _viewer_is_running(viewer):
                break

            panel.update()
            if panel.should_quit():
                print("[INFO] panel requested quit.")
                break

            action = panel.action()
            unsupported = sorted(set(action) - ALLOWED_ACTION_KEYS)
            if unsupported:
                print(f"[FAIL] unsupported action keys: {unsupported}")
                return 5

            paused = _is_paused(panel)
            if not paused:
                try:
                    _apply_feagine_action(robot, action)
                except ValueError as exc:
                    if "unsupported action keys" in str(exc):
                        print(f"[FAIL] {exc}")
                        return 5
                    print(
                        "[FAIL] Could not apply Feagine action to tabletop scene. "
                        "Need shared action applier in a later step."
                    )
                    print(f"[DETAIL] {exc}")
                    return 4
                except Exception as exc:
                    print(
                        "[FAIL] Could not apply Feagine action to tabletop scene. "
                        "Need shared action applier in a later step."
                    )
                    print(f"[DETAIL] {exc}")
                    return 4
                mujoco.mj_step(model, data)

            tip_position = _body_position(mujoco, model, data, "feagine_grasper_tip")
            red_position = _body_position(mujoco, model, data, "red_object")
            blue_position = _body_position(mujoco, model, data, "blue_object")
            obstacle_position = _body_position(mujoco, model, data, "black_obstacle")
            target_position = _body_position(mujoco, model, data, "target_pad")
            panel.set_tip_position(tip_position)

            if not _viewer_sync(viewer):
                break

            if step_index % print_every == 0:
                print(
                    f"[STEP {step_index:04d}] paused={paused} "
                    f"section_angles={action['section_angles']} "
                    f"grip={action['grip_command']} "
                    f"rotation={action['grasper_rotation']} "
                    f"tip={tip_position} "
                    f"red={red_position} "
                    f"blue={blue_position} "
                    f"obstacle={obstacle_position} "
                    f"target={target_position}"
                )

            _sleep(float(args.realtime_factor))
    except KeyboardInterrupt:
        pass
    finally:
        if panel is not None:
            panel.close()
        if viewer is not None:
            _close_viewer(viewer)

    print("[OK] tabletop panel UI finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
