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

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop smoke test for calibrated PccIkController.")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--target-offset", nargs=3, type=float, default=[0.0, 0.02, 0.0])
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--success-threshold", type=float, default=0.005)
    parser.add_argument("--output", default="outputs/diagnostics/pcc_ik_closed_loop_y_plus.json")
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
    tip_pose = robot_state.get("tip_pose")
    if not isinstance(tip_pose, dict) or "position" not in tip_pose:
        raise ValueError("robot_state is missing tip_pose.position")
    return robot_state


def _tip_position(observation: dict[str, Any]) -> np.ndarray:
    return _array(_robot_state(observation)["tip_pose"]["position"])[:3]


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


def main() -> int:
    args = _parse_args()
    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        print(f"[FAIL] calibration file does not exist: {calibration_path}")
        return 1

    try:
        env = _make_env(args.max_steps)
        controller = PccIkController(calibration_path=calibration_path)
        observation = env.reset()
    except Exception as exc:
        print(f"[FAIL] initialization failed: {exc}")
        return 1

    try:
        initial_tip = _tip_position(observation)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2
    target_offset = _array(args.target_offset)[:3]
    target_tip = initial_tip + target_offset
    initial_distance = _norm(target_tip - initial_tip)

    print(f"[INFO] initial_tip={initial_tip.tolist()}")
    print(f"[INFO] target_tip={target_tip.tolist()}")
    print(f"[INFO] initial_distance={initial_distance:.9f}")

    step_logs: list[dict[str, Any]] = []
    final_observation = observation
    done = False
    for step_index in range(int(args.max_steps)):
        try:
            robot_state = _robot_state(final_observation)
        except ValueError as exc:
            print(f"[FAIL] {exc}")
            return 2
        target_state = {
            "target_tip_position": target_tip.tolist(),
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        }
        action, ik_info = controller.compute_action_with_info(target_state, robot_state)
        if "gripper_rotation" in action:
            print("[FAIL] action contains unsupported gripper_rotation key.")
            return 2
        try:
            final_observation, _reward, done, _info = _step_env(env, action)
        except Exception as exc:
            print(f"[FAIL] env.step failed at step {step_index}: {exc}")
            return 1

        tip = _tip_position(final_observation)
        distance = _norm(target_tip - tip)
        y_error = float(target_tip[1] - tip[1])
        current_robot_state = _robot_state(final_observation)
        step_log = {
            "step": int(step_index),
            "tip_position": tip.tolist(),
            "target_tip_position": target_tip.tolist(),
            "distance_to_target": distance,
            "y_error": y_error,
            "section_angles": _jsonable(current_robot_state.get("section_angles", [])),
            "action": _jsonable(action),
            "ik_info": _jsonable(ik_info),
            "qpos_norm": _norm(current_robot_state.get("qpos", [])),
            "ctrl_norm": _norm(current_robot_state.get("ctrl", [])),
            "contact": _jsonable(final_observation.get("contact", {})),
        }
        step_logs.append(step_log)
        print(
            f"[STEP {step_index:03d}] distance={distance:.9f} "
            f"y={float(tip[1]):.9f} action_section_angles={action['section_angles']}"
        )
        if distance < float(args.success_threshold):
            break
        if done:
            break

    final_tip = _tip_position(final_observation)
    final_distance = _norm(target_tip - final_tip)
    distance_reduction = float(initial_distance - final_distance)
    initial_y = float(initial_tip[1])
    final_y = float(final_tip[1])
    target_y = float(target_tip[1])
    y_progress = float(final_y - initial_y)
    final_robot_state = _robot_state(final_observation)
    initial_section_angles = _array(_robot_state(observation).get("section_angles", []))
    final_section_angles = _array(final_robot_state.get("section_angles", []))
    section_angle_change_norm = _norm(final_section_angles - initial_section_angles)
    max_abs_section_angle = float(np.max(np.abs(final_section_angles))) if final_section_angles.size else 0.0
    success = final_distance < float(args.success_threshold)
    improved = final_distance < initial_distance

    print(f"[RESULT] final_tip={final_tip.tolist()}")
    print(f"[RESULT] final_distance={final_distance:.9f}")
    print(f"[RESULT] distance_reduction={distance_reduction:.9f}")
    print(f"[RESULT] y_progress={y_progress:.9f}")
    print(f"[RESULT] success={success}")
    print(f"[RESULT] improved={improved}")

    if improved:
        judgment = "[OK] PccIkController improves distance to the +y target in closed loop."
    elif y_progress > 0.0:
        judgment = "[WARN] tip moves in +y but total distance does not improve."
    else:
        judgment = "[FAIL] tip does not move toward +y target."
    print(judgment)

    payload = {
        "calibration": str(calibration_path),
        "target_offset": target_offset.tolist(),
        "initial_tip_position": initial_tip.tolist(),
        "target_tip_position": target_tip.tolist(),
        "final_tip_position": final_tip.tolist(),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "distance_reduction": distance_reduction,
        "initial_y": initial_y,
        "final_y": final_y,
        "target_y": target_y,
        "y_progress": y_progress,
        "section_angle_change_norm": section_angle_change_norm,
        "max_abs_section_angle": max_abs_section_angle,
        "success_threshold": float(args.success_threshold),
        "success": success,
        "improved": improved,
        "num_steps_executed": len(step_logs),
        "step_logs": step_logs,
        "judgment": judgment,
    }
    try:
        _write_json(args.output, payload)
    except Exception as exc:
        print(f"[FAIL] could not write JSON: {exc}")
        return 3
    print(f"[OK] wrote {args.output}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
