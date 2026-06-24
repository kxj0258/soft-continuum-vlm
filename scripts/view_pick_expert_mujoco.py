from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.scripted_expert import ScriptedExpert
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View the virtual pick_red_object expert in MuJoCo.")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--target-offset", nargs=3, type=float, default=[0.0, 0.02, 0.0])
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--realtime-factor", type=float, default=0.5)
    parser.add_argument("--hold-view-steps", type=int, default=30)
    parser.add_argument("--close-hold-position-gain-scale", type=float, default=0.35)
    parser.add_argument("--max-post-close-drift", type=float, default=0.02)
    parser.add_argument("--done-hold-mode", choices=["closed_loop", "open_loop"], default="closed_loop")
    parser.add_argument("--done-hold-position-gain-scale", type=float, default=0.20)
    parser.add_argument("--done-max-section-step-norm", type=float, default=0.03)
    parser.add_argument("--done-overshoot-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--done-overshoot-margin", type=float, default=0.003)
    return parser.parse_args()


def _make_env(max_steps: int, hold_view_steps: int) -> FeagineMujocoEnv:
    return FeagineMujocoEnv(
        {
            "env": {
                "robot_preset": "a03_type_2",
                "asset_model_type": "mjcf",
                "render_mode": "none",
                "max_episode_steps": max(int(max_steps) + int(hold_view_steps) + 1, 2),
            }
        }
    )


def _robot_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    robot_state = observation.get("robot_state")
    return dict(robot_state) if isinstance(robot_state, Mapping) else {}


def _tip_position(observation: Mapping[str, Any]) -> list[float]:
    tip_pose = _robot_state(observation).get("tip_pose")
    if isinstance(tip_pose, Mapping) and "position" in tip_pose:
        return _vector(tip_pose["position"], length=3)
    return [0.0, 0.0, 0.0]


def _vector(value: Any, *, length: int | None = None) -> list[float]:
    array = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    if length is None:
        return [float(item) for item in array.tolist()]
    result = np.zeros(length, dtype=np.float64)
    copy_count = min(length, array.size)
    if copy_count:
        result[:copy_count] = array[:copy_count]
    return [float(item) for item in result.tolist()]


def _finite_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _step_env(env: FeagineMujocoEnv, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 4:
        observation, reward, done, info = result
        return dict(observation), float(reward), bool(done), dict(info)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return dict(observation), float(reward), bool(terminated or truncated), dict(info)
    raise ValueError(f"env.step returned {len(result)} values; expected 4 or 5")


def _phase_sequence(phases: list[str]) -> list[str]:
    sequence: list[str] = []
    for phase in phases:
        if not sequence or sequence[-1] != phase:
            sequence.append(phase)
    return sequence


def _model_data(env: FeagineMujocoEnv) -> tuple[Any | None, Any | None]:
    model = None
    data = None
    for name in ("model", "mj_model", "mujoco_model", "_model"):
        model = getattr(env, name, None)
        if model is not None:
            break
    for name in ("data", "mj_data", "mujoco_data", "_data"):
        data = getattr(env, name, None)
        if data is not None:
            break
    return model, data


def _viewer_is_running(viewer: Any) -> bool:
    is_running = getattr(viewer, "is_running", None)
    if callable(is_running):
        try:
            return bool(is_running())
        except Exception:
            return False
    return True


def _viewer_sync(viewer: Any) -> bool:
    try:
        viewer.sync()
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


def _sleep_duration(model: Any, env: FeagineMujocoEnv, realtime_factor: float) -> float:
    if realtime_factor <= 0.0:
        return 0.0
    timestep = getattr(getattr(model, "opt", None), "timestep", None)
    if timestep is None:
        env_dt = 0.03
    else:
        physics_steps = int(getattr(getattr(env, "config", None), "physics_steps_per_action", 1))
        env_dt = float(timestep) * max(physics_steps, 1)
    return max(0.0, env_dt / float(realtime_factor))


def _print_summary(
    *,
    phases: list[str],
    best_distance: float,
    final_distance: float,
    post_close_drift: float,
    final_grip_command: float,
    done_hold_mode: str,
    done_overshoot_guard_triggered_count: int,
    max_post_done_distance: float,
    final_y_minus_target_y: float,
) -> None:
    print(f"[RESULT] phase_sequence={_phase_sequence(phases)}")
    print(f"[RESULT] best_distance={best_distance:.9f}")
    print(f"[RESULT] final_distance={final_distance:.9f}")
    print(f"[RESULT] post_close_drift={post_close_drift:.9f}")
    print(f"[RESULT] final_grip_command={final_grip_command:.6f}")
    print(f"[RESULT] done_hold_mode={done_hold_mode}")
    print(f"[RESULT] done_overshoot_guard_triggered_count={done_overshoot_guard_triggered_count}")
    print(f"[RESULT] max_post_done_distance={max_post_done_distance:.9f}")
    print(f"[RESULT] final_y_minus_target_y={final_y_minus_target_y:.9f}")
    print("[OK] viewer run finished")


def main() -> int:
    args = _parse_args()
    env: FeagineMujocoEnv | None = None
    viewer: Any | None = None
    try:
        env = _make_env(args.max_steps, args.hold_view_steps)
        obs = dict(env.reset())
        expert = ScriptedExpert(
            task_name="pick_red_object",
            calibration_path=args.calibration,
            target_offset=args.target_offset,
            close_loop_hold=True,
            close_hold_position_gain_scale=args.close_hold_position_gain_scale,
            max_post_close_drift=args.max_post_close_drift,
            done_hold_mode=args.done_hold_mode,
            done_hold_position_gain_scale=args.done_hold_position_gain_scale,
            done_max_section_step_norm=args.done_max_section_step_norm,
            done_overshoot_guard=args.done_overshoot_guard,
            done_overshoot_margin=args.done_overshoot_margin,
        )
        expert.reset(task_name="pick_red_object", language=str(obs.get("language", "")))
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
        print("[FAIL] Failed to open MuJoCo viewer. If running headless, use an offscreen render script instead.")
        print(f"[DETAIL] {exc}")
        env.close()
        return 1

    phases: list[str] = []
    best_distance = float("inf")
    final_distance = float("nan")
    post_close_drift = float("nan")
    final_grip_command = 0.0
    done_hold_mode = str(args.done_hold_mode)
    done_overshoot_guard_triggered_count = 0
    max_post_done_distance = float("nan")
    final_y_minus_target_y = float("nan")
    hold_remaining: int | None = None
    sleep_seconds = _sleep_duration(model, env, float(args.realtime_factor))

    try:
        for step_index in range(int(args.max_steps) + int(args.hold_view_steps) + 1):
            if not _viewer_is_running(viewer):
                break

            state = _robot_state(obs)
            tip = _tip_position(obs)
            _section_angles = _vector(state.get("section_angles", []))
            _grasper_rotation = _finite_float(state.get("grasper_rotation"), 0.0)
            action, expert_info = expert.act(obs)
            if "gripper_rotation" in action:
                print("[FAIL] action contains unsupported gripper_rotation key.")
                return 1

            phase = str(expert_info.get("phase", "unknown"))
            phases.append(phase)
            obs, _reward, env_done, _env_info = _step_env(env, action)
            if not _viewer_sync(viewer):
                break

            distance = _finite_float(
                expert_info.get("current_distance_to_target", expert_info.get("target_distance")),
                float("nan"),
            )
            best_from_info = _finite_float(expert_info.get("best_distance_to_target"), distance)
            if np.isfinite(distance):
                best_distance = min(best_distance, distance)
                final_distance = distance
            if np.isfinite(best_from_info):
                best_distance = min(best_distance, best_from_info)
            if phase == "done" and np.isfinite(distance):
                max_post_done_distance = (
                    distance
                    if not np.isfinite(max_post_done_distance)
                    else max(max_post_done_distance, distance)
                )
            if bool(expert_info.get("done_overshoot_guard_triggered", False)):
                done_overshoot_guard_triggered_count += 1
            done_hold_mode = str(expert_info.get("done_hold_mode", done_hold_mode))
            final_y_minus_target_y = _finite_float(
                expert_info.get("current_y_minus_target_y"),
                final_y_minus_target_y,
            )
            post_close_drift = _finite_float(
                expert_info.get("post_close_drift"),
                final_distance - best_distance if np.isfinite(final_distance) and np.isfinite(best_distance) else float("nan"),
            )
            final_grip_command = _finite_float(action.get("grip_command", state.get("grip_command")), 0.0)
            print(
                f"[STEP {step_index:03d}] phase={phase} distance={distance:.4f} "
                f"best={best_distance:.4f} grip={final_grip_command:.2f} tip={tip}"
            )

            if phase == "done":
                if hold_remaining is None:
                    hold_remaining = int(args.hold_view_steps)
                elif hold_remaining <= 0:
                    break
                else:
                    hold_remaining -= 1
            elif env_done:
                break

            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[FAIL] viewer loop failed: {exc}")
        return 1
    finally:
        if viewer is not None:
            _close_viewer(viewer)
        if env is not None:
            env.close()

    if not np.isfinite(best_distance):
        best_distance = float("nan")
    final_state = _robot_state(obs)
    final_grip_command = _finite_float(final_state.get("grip_command"), final_grip_command)
    if np.isfinite(final_distance) and np.isfinite(best_distance) and not np.isfinite(post_close_drift):
        post_close_drift = final_distance - best_distance
    _print_summary(
        phases=phases,
        best_distance=best_distance,
        final_distance=final_distance,
        post_close_drift=post_close_drift,
        final_grip_command=final_grip_command,
        done_hold_mode=done_hold_mode,
        done_overshoot_guard_triggered_count=done_overshoot_guard_triggered_count,
        max_post_done_distance=max_post_done_distance,
        final_y_minus_target_y=final_y_minus_target_y,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
