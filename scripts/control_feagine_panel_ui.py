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
from soft_continuum_vlm.ui.tk_control_panel import TkFeagineControlPanel


ALLOWED_ACTION_KEYS = {"section_angles", "grip_command", "grasper_rotation"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control the Feagine MuJoCo arm with a Tkinter control panel."
    )
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--panel-title", type=str, default="Feagine Panel Control")
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


def _step_env(
    env: FeagineMujocoEnv, action: dict[str, Any]
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    step_result = env.step(action)
    if len(step_result) == 4:
        obs, reward, done, info = step_result
        return dict(obs), float(reward), bool(done), dict(info)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return dict(obs), float(reward), bool(terminated or truncated), dict(info)
    raise ValueError(f"env.step returned {len(step_result)} values; expected 4 or 5.")


def _model_data(env: FeagineMujocoEnv) -> tuple[Any | None, Any | None]:
    model = getattr(env, "model", None) or getattr(env, "_model", None)
    data = getattr(env, "data", None) or getattr(env, "_data", None)
    return model, data


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
    env: FeagineMujocoEnv | None = None
    panel: TkFeagineControlPanel | None = None
    viewer: Any | None = None
    obs: dict[str, Any] = {}
    print_every = max(1, int(args.print_every))

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

        viewer = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        print("[FAIL] Failed to open MuJoCo viewer.")
        print(f"[DETAIL] {exc}")
        env.close()
        return 1

    try:
        panel = TkFeagineControlPanel(title=args.panel_title)
    except Exception:
        print("[FAIL] Failed to open Tkinter control panel.")
        _close_viewer(viewer)
        env.close()
        return 2

    panel.set_tip_position(_tip_position(obs))
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
                return 3

            paused = _is_paused(panel)
            if not paused:
                obs, _reward, done, _info = _step_env(env, action)
                if done:
                    print("[INFO] env done; resetting.")
                    obs = _reset_env(env)

            panel.set_tip_position(_tip_position(obs))
            if not _viewer_sync(viewer):
                break

            if step_index % print_every == 0:
                print(
                    f"[STEP {step_index:04d}] paused={paused} "
                    f"section_angles={action['section_angles']} "
                    f"grip={action['grip_command']} "
                    f"rotation={action['grasper_rotation']} "
                    f"tip={_tip_position(obs)}"
                )

            _sleep(float(args.realtime_factor))
    except KeyboardInterrupt:
        pass
    finally:
        if panel is not None:
            panel.close()
        if viewer is not None:
            _close_viewer(viewer)
        if env is not None:
            env.close()

    print("[OK] Feagine panel UI finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
