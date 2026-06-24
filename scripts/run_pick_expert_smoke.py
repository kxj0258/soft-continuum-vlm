from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.scripted_expert import ScriptedExpert
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the virtual pick_red_object scripted expert.")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--target-offset", nargs=3, type=float, default=[0.0, 0.02, 0.0])
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--close-loop-hold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--close-hold-position-gain-scale", type=float, default=0.35)
    parser.add_argument("--max-post-close-drift", type=float, default=0.02)
    parser.add_argument("--done-hold-mode", choices=["closed_loop", "open_loop"], default="closed_loop")
    parser.add_argument("--done-hold-position-gain-scale", type=float, default=0.20)
    parser.add_argument("--done-max-section-step-norm", type=float, default=0.03)
    parser.add_argument("--done-overshoot-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--done-overshoot-margin", type=float, default=0.003)
    parser.add_argument("--output", default="outputs/diagnostics/pick_expert_smoke.json")
    return parser.parse_args()


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _norm(value: Any) -> float:
    return float(np.linalg.norm(_array(value)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _robot_state(observation: dict[str, Any]) -> dict[str, Any]:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        raise ValueError("observation is missing robot_state")
    return robot_state


def _tip(observation: dict[str, Any]) -> list[float]:
    robot_state = _robot_state(observation)
    tip_pose = robot_state.get("tip_pose")
    if not isinstance(tip_pose, dict) or "position" not in tip_pose:
        raise ValueError("robot_state is missing tip_pose.position")
    return _array(tip_pose["position"])[:3].tolist()


def _step_env(env: FeagineMujocoEnv, action: dict[str, Any]) -> tuple[dict[str, Any], Any, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, reward, bool(done), dict(info)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated or truncated), dict(info)
    raise ValueError(f"env.step returned {len(result)} values; expected 4 or 5")


def _make_env(max_steps: int) -> FeagineMujocoEnv:
    return FeagineMujocoEnv(
        {
            "env": {
                "robot_preset": "a03_type_2",
                "asset_model_type": "mjcf",
                "render_mode": "none",
                "max_episode_steps": max(int(max_steps) + 1, 2),
            }
        }
    )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _phase_sequence(phases: list[str]) -> list[str]:
    sequence: list[str] = []
    for phase in phases:
        if not sequence or sequence[-1] != phase:
            sequence.append(phase)
    return sequence


def main() -> int:
    args = _parse_args()
    try:
        env = _make_env(args.max_steps)
        observation = env.reset()
        expert = ScriptedExpert(
            task_name="pick_red_object",
            calibration_path=args.calibration,
            target_offset=args.target_offset,
            close_loop_hold=args.close_loop_hold,
            close_hold_position_gain_scale=args.close_hold_position_gain_scale,
            max_post_close_drift=args.max_post_close_drift,
            done_hold_mode=args.done_hold_mode,
            done_hold_position_gain_scale=args.done_hold_position_gain_scale,
            done_max_section_step_norm=args.done_max_section_step_norm,
            done_overshoot_guard=args.done_overshoot_guard,
            done_overshoot_margin=args.done_overshoot_margin,
        )
        initial_tip = _tip(observation)
    except Exception as exc:
        print(f"[FAIL] initialization failed: {exc}")
        return 1

    step_logs: list[dict[str, Any]] = []
    phases: list[str] = []
    virtual_red_object_position: list[float] | None = None
    final_observation = observation
    done = False
    for step_index in range(int(args.max_steps)):
        action, expert_info = expert.act(final_observation)
        if "gripper_rotation" in action:
            print("[FAIL] action contains unsupported gripper_rotation key.")
            return 1
        try:
            final_observation, reward, done, env_info = _step_env(env, action)
            tip = _tip(final_observation)
        except Exception as exc:
            print(f"[FAIL] step {step_index} failed: {exc}")
            return 1

        phase = str(expert_info.get("phase", "unknown"))
        phases.append(phase)
        if virtual_red_object_position is None:
            virtual_red_object_position = list(expert_info.get("red_object_position", []))
        target_distance = float(expert_info.get("target_distance", 0.0))
        grip = float(action.get("grip_command", 0.0))
        print(f"[STEP {step_index:03d}] phase={phase} target_distance={target_distance:.9f} grip={grip:.3f} tip={tip}")
        step_logs.append(
            {
                "step": int(step_index),
                "phase": phase,
                "target_distance": target_distance,
                "current_distance_to_target": expert_info.get("current_distance_to_target"),
                "best_distance_to_target": expert_info.get("best_distance_to_target"),
                "post_close_drift": expert_info.get("post_close_drift"),
                "overshoot_guard_triggered": bool(expert_info.get("overshoot_guard_triggered", False)),
                "done_overshoot_guard_triggered": bool(
                    expert_info.get("done_overshoot_guard_triggered", False)
                ),
                "done_hold_mode": expert_info.get("done_hold_mode"),
                "current_y_minus_target_y": expert_info.get("current_y_minus_target_y"),
                "drift_warning": bool(expert_info.get("drift_warning", False)),
                "tip": tip,
                "action": _jsonable(action),
                "expert_info": _jsonable(expert_info),
                "reward": float(reward),
                "done": bool(done),
                "env_info": _jsonable(env_info),
            }
        )
        if phase == "done" or done:
            break

    if virtual_red_object_position is None:
        virtual_red_object_position = (_array(initial_tip) + _array(args.target_offset)[:3]).tolist()
    final_tip = _tip(final_observation)
    initial_distance = _norm(_array(virtual_red_object_position) - _array(initial_tip))
    final_distance = _norm(_array(virtual_red_object_position) - _array(final_tip))
    tip_history = [initial_tip] + [list(log["tip"]) for log in step_logs]
    distances = [_norm(_array(virtual_red_object_position) - _array(tip)) for tip in tip_history]
    best_distance = float(min(distances)) if distances else float(initial_distance)
    best_distance_reduction = float(initial_distance - best_distance)
    final_distance_reduction = float(initial_distance - final_distance)
    post_close_drift = float(final_distance - best_distance)
    final_robot_state = _robot_state(final_observation)
    final_grip_command = float(final_robot_state.get("grip_command", 0.0))
    phase_sequence = _phase_sequence(phases)
    reached_close_phase = "close_gripper" in phases
    reached_done_phase = "done" in phases
    position_tolerance = float(getattr(expert, "position_tolerance", 0.006))
    overshoot_guard_count = sum(1 for log in step_logs if log["overshoot_guard_triggered"])
    done_overshoot_guard_count = sum(1 for log in step_logs if log["done_overshoot_guard_triggered"])
    drift_warning_count = sum(1 for log in step_logs if log["drift_warning"])
    post_done_distances = [
        float(log["current_distance_to_target"])
        for log in step_logs
        if log["phase"] == "done" and log.get("current_distance_to_target") is not None
    ]
    max_post_done_distance = float(max(post_done_distances)) if post_done_distances else float("nan")
    final_y_minus_target_y = float(_array(final_tip)[1] - _array(virtual_red_object_position)[1])

    if (
        best_distance < position_tolerance
        and reached_close_phase
        and final_grip_command >= 0.9
        and post_close_drift < float(args.max_post_close_drift)
    ):
        judgment = "[OK] pick_red_object expert reaches virtual target, closes gripper, and holds without large drift."
    elif best_distance < position_tolerance and reached_close_phase and final_grip_command >= 0.9:
        judgment = "[WARN] expert approaches target and closes gripper, but hold drift is large."
    else:
        judgment = "[FAIL] expert does not reliably approach the virtual pick target."
    print(f"[RESULT] initial_tip={initial_tip}")
    print(f"[RESULT] virtual_red_object_position={virtual_red_object_position}")
    print(f"[RESULT] final_tip={final_tip}")
    print(f"[RESULT] initial_distance_to_virtual_target={initial_distance:.9f}")
    print(f"[RESULT] best_distance_to_virtual_target={best_distance:.9f}")
    print(f"[RESULT] final_distance_to_virtual_target={final_distance:.9f}")
    print(f"[RESULT] best_distance_reduction={best_distance_reduction:.9f}")
    print(f"[RESULT] final_distance_reduction={final_distance_reduction:.9f}")
    print(f"[RESULT] post_close_drift={post_close_drift:.9f}")
    print(f"[RESULT] final_grip_command={final_grip_command:.6f}")
    print(f"[RESULT] phase_sequence={phase_sequence}")
    print(f"[RESULT] reached_close_phase={reached_close_phase}")
    print(f"[RESULT] reached_done_phase={reached_done_phase}")
    print(f"[RESULT] overshoot_guard_triggered_count={overshoot_guard_count}")
    print(f"[RESULT] done_hold_mode={args.done_hold_mode}")
    print(f"[RESULT] done_overshoot_guard_triggered_count={done_overshoot_guard_count}")
    print(f"[RESULT] max_post_done_distance={max_post_done_distance:.9f}")
    print(f"[RESULT] final_y_minus_target_y={final_y_minus_target_y:.9f}")
    print(f"[RESULT] drift_warning_count={drift_warning_count}")
    print(judgment)

    payload = {
        "initial_tip": initial_tip,
        "virtual_red_object_position": virtual_red_object_position,
        "final_tip": final_tip,
        "initial_distance_to_virtual_target": initial_distance,
        "best_distance_to_virtual_target": best_distance,
        "final_distance_to_virtual_target": final_distance,
        "best_distance_reduction": best_distance_reduction,
        "final_distance_reduction": final_distance_reduction,
        "post_close_drift": post_close_drift,
        "final_grip_command": final_grip_command,
        "phase_sequence": phase_sequence,
        "phase_history": phases,
        "reached_close_phase": reached_close_phase,
        "reached_done_phase": reached_done_phase,
        "overshoot_guard_triggered_count": overshoot_guard_count,
        "done_hold_mode": args.done_hold_mode,
        "done_overshoot_guard_triggered_count": done_overshoot_guard_count,
        "max_post_done_distance": max_post_done_distance,
        "final_y_minus_target_y": final_y_minus_target_y,
        "drift_warning_count": drift_warning_count,
        "judgment": judgment,
        "step_logs": step_logs,
    }
    try:
        _write_json(args.output, payload)
    except Exception as exc:
        print(f"[FAIL] could not write JSON: {exc}")
        return 1
    print(f"[OK] wrote {args.output}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
