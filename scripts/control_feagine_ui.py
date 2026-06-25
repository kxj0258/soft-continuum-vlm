from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv
from soft_continuum_vlm.ui.manual_control import ManualControlConfig, ManualFeagineController


SUPPORTED_ACTION_KEYS = {"section_angles", "grip_command", "grasper_rotation"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control the Feagine MuJoCo arm with keyboard input.")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--section-step", type=float, default=0.03)
    parser.add_argument("--grasper-rotation-step", type=float, default=0.05)
    parser.add_argument("--max-abs-section-angle", type=float, default=0.8)
    return parser.parse_args()


def _make_env(max_steps: int) -> FeagineMujocoEnv:
    return FeagineMujocoEnv(
        {
            "env": {
                "robot_preset": "a03_type_2",
                "asset_model_type": "mjcf",
                "render_mode": "none",
                "max_episode_steps": int(max_steps),
            }
        }
    )


def _reset_env(env: FeagineMujocoEnv) -> dict[str, Any]:
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        return dict(reset_result[0])
    return dict(reset_result)


def _step_env(env: FeagineMujocoEnv, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    step_result = env.step(action)
    if len(step_result) == 4:
        obs, reward, done, info = step_result
        return dict(obs), float(reward), bool(done), dict(info)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return dict(obs), float(reward), bool(terminated or truncated), dict(info)
    raise ValueError(f"env.step returned {len(step_result)} values; expected 4 or 5.")


def _model_data(env: FeagineMujocoEnv) -> tuple[Any | None, Any | None]:
    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    if model is None:
        model = getattr(env, "_model", None)
    if data is None:
        data = getattr(env, "_data", None)
    return model, data


def key_to_name(key: int) -> str:
    key_code = int(key)
    try:
        import glfw  # type: ignore
    except Exception:
        glfw = None

    if glfw is not None:
        if key_code == int(glfw.KEY_SPACE):
            return "space"
        if key_code == int(glfw.KEY_ESCAPE):
            return "esc"

    if key_code == 32:
        return "space"
    if key_code == 27 or key_code == 256:
        return "esc"
    if 32 <= key_code <= 126:
        return chr(key_code).lower()
    return ""


def _tip_position(obs: Mapping[str, Any]) -> Any:
    robot_state = obs.get("robot_state")
    if not isinstance(robot_state, Mapping):
        return None
    tip_pose = robot_state.get("tip_pose")
    if not isinstance(tip_pose, Mapping):
        return None
    return tip_pose.get("position")


def _viewer_is_running(viewer: Any) -> bool:
    is_running = getattr(viewer, "is_running", None)
    if callable(is_running):
        try:
            return bool(is_running())
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


def _print_keys() -> None:
    print("[KEYS]")
    print("1/q: section 0 +/-")
    print("2/w: section 2 +/-")
    print("3/e: section 4 +/-")
    print("a/z: section 1 +/-")
    print("s/x: section 3 +/-")
    print("d/c: section 5 +/-")
    print("o/p: open/close gripper")
    print("r/f: grasper rotation +/-")
    print("space: pause/resume")
    print("0: reset command")
    print("esc: quit")


def _sleep(realtime_factor: float) -> None:
    if realtime_factor <= 0.0:
        return
    time.sleep(max(0.0, 0.03 / float(realtime_factor)))


def main() -> int:
    args = _parse_args()
    controller = ManualFeagineController(
        ManualControlConfig(
            section_step=args.section_step,
            grasper_rotation_step=args.grasper_rotation_step,
            max_abs_section_angle=args.max_abs_section_angle,
        )
    )
    env: FeagineMujocoEnv | None = None
    viewer: Any | None = None

    def key_callback(key: int) -> None:
        key_name = key_to_name(key)
        if key_name:
            controller.handle_key(key_name)

    try:
        env = _make_env(args.max_steps)
        obs = _reset_env(env)
    except Exception as exc:
        print(f"[FAIL] initialization failed: {exc}")
        return 1

    model, data = _model_data(env)
    if model is None or data is None:
        print("[FAIL] FeagineMujocoEnv does not expose model/data for MuJoCo viewer.")
        env.close()
        return 1

    try:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    except TypeError as exc:
        print(
            "[FAIL] This MuJoCo viewer version does not support key_callback. "
            "Please use a MuJoCo version with launch_passive(..., key_callback=...)."
        )
        print(f"[DETAIL] {exc}")
        env.close()
        return 1
    except Exception as exc:
        print("[FAIL] Failed to open MuJoCo viewer.")
        print(f"[DETAIL] {exc}")
        env.close()
        return 1

    _print_keys()
    print_every = max(1, int(args.print_every))

    try:
        for step_index in range(int(args.max_steps)):
            if controller.state.quit_requested or not _viewer_is_running(viewer):
                break

            if controller.state.paused:
                viewer.sync()
                _sleep(float(args.realtime_factor))
                continue

            action = controller.action()
            unsupported = sorted(set(action) - SUPPORTED_ACTION_KEYS)
            if unsupported:
                print(f"[FAIL] unsupported action key(s): {unsupported}")
                return 1

            obs, _reward, done, _info = _step_env(env, action)
            viewer.sync()

            if step_index % print_every == 0:
                print(
                    f"[STEP {step_index:04d}] paused={controller.state.paused} "
                    f"section_angles={action['section_angles']} "
                    f"grip={action['grip_command']} "
                    f"rotation={action['grasper_rotation']} "
                    f"tip={_tip_position(obs)}"
                )

            if done:
                print("[INFO] env done; resetting.")
                obs = _reset_env(env)

            _sleep(float(args.realtime_factor))
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            _close_viewer(viewer)
        if env is not None:
            env.close()

    print("[OK] manual Feagine UI finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
