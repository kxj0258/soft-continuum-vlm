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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Feagine section_angles and record model response.")
    parser.add_argument("--amplitude", type=float, default=0.3)
    parser.add_argument("--steps-per-command", type=int, default=20)
    parser.add_argument("--output", default="outputs/diagnostics/section_angle_sweep.json")
    return parser.parse_args()


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _norm(value: Any) -> float:
    return float(np.linalg.norm(_as_array(value)))


def _snapshot(observation: dict[str, Any]) -> dict[str, Any] | None:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        return None
    tip_pose = robot_state.get("tip_pose")
    if not isinstance(tip_pose, dict):
        return None
    return {
        "tip_position": [float(value) for value in tip_pose.get("position", [0.0, 0.0, 0.0])],
        "qpos": _as_array(robot_state.get("qpos", [])).tolist(),
        "qvel": _as_array(robot_state.get("qvel", [])).tolist(),
        "ctrl": _as_array(robot_state.get("ctrl", [])).tolist(),
        "section_angles": _as_array(robot_state.get("section_angles", [])).tolist(),
    }


def _section_angle_length(observation: dict[str, Any]) -> int | None:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        return None
    tip_pose = robot_state.get("tip_pose")
    if not isinstance(tip_pose, dict):
        return None
    return len(robot_state.get("section_angles", []) or [])


def _tip_source(observation: dict[str, Any]) -> str:
    robot_state = observation.get("robot_state", {})
    if not isinstance(robot_state, dict):
        return "unresolved"
    tip_pose = robot_state.get("tip_pose", {})
    if not isinstance(tip_pose, dict):
        return "unresolved"
    return str(tip_pose.get("source", "unresolved"))


def _fit_command(values: list[float], length: int) -> list[float]:
    if len(values) >= length:
        return [float(value) for value in values[:length]]
    return [float(value) for value in values] + [0.0] * (length - len(values))


def _base_commands(amplitude: float) -> dict[str, list[float]]:
    amp = float(amplitude)
    return {
        "zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "plus_x_all": [amp, 0.0, amp, 0.0, amp, 0.0],
        "minus_x_all": [-amp, 0.0, -amp, 0.0, -amp, 0.0],
        "plus_y_all": [0.0, amp, 0.0, amp, 0.0, amp],
        "minus_y_all": [0.0, -amp, 0.0, -amp, 0.0, -amp],
        "x_gradient": [amp, 0.0, amp / 2.0, 0.0, -amp / 2.0, 0.0],
        "y_gradient": [0.0, amp, 0.0, amp / 2.0, 0.0, -amp / 2.0],
    }


def _build_env(steps_per_command: int) -> Any:
    from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv

    return FeagineMujocoEnv(
        {
            "env": {
                "robot_preset": "a03_type_2",
                "asset_model_type": "mjcf",
                "render_mode": "none",
                "max_episode_steps": max(int(steps_per_command) + 1, 2),
            }
        }
    )


def _run_command(env: Any, command: list[float], steps_per_command: int) -> dict[str, Any] | None:
    before_observation = env.reset()
    before = _snapshot(before_observation)
    if before is None:
        return None

    action = {
        "section_angles": command,
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    }
    tip_trace: list[list[float]] = []
    after_observation = before_observation
    for _ in range(steps_per_command):
        after_observation, _reward, _done, _info = env.step(action)
        step_snapshot = _snapshot(after_observation)
        if step_snapshot is None:
            return None
        tip_trace.append(step_snapshot["tip_position"])

    after = _snapshot(after_observation)
    if after is None:
        return None

    before_tip = _as_array(before["tip_position"])
    after_tip = _as_array(after["tip_position"])
    delta_tip = after_tip - before_tip
    return {
        "command": command,
        "before": before,
        "after": after,
        "delta_tip": delta_tip.tolist(),
        "delta_tip_norm": _norm(delta_tip),
        "delta_qpos_norm": _norm(_as_array(after["qpos"]) - _as_array(before["qpos"])),
        "delta_qvel_norm": _norm(_as_array(after["qvel"]) - _as_array(before["qvel"])),
        "delta_ctrl_norm": _norm(_as_array(after["ctrl"]) - _as_array(before["ctrl"])),
        "delta_section_angles_norm": _norm(
            _as_array(after["section_angles"]) - _as_array(before["section_angles"])
        ),
        "ctrl_norm_after": _norm(after["ctrl"]),
        "qpos_norm_after": _norm(after["qpos"]),
        "tip_trace": tip_trace,
    }


def _write_json(output: str, payload: dict[str, Any]) -> bool:
    output_path = Path(output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[FAIL] could not write JSON: {exc}")
        return False
    return True


def main() -> int:
    args = _parse_args()
    try:
        env = _build_env(args.steps_per_command)
    except Exception as exc:
        print(f"[FAIL] environment creation failed: {exc}")
        return 1

    try:
        initial_observation = env.reset()
    except Exception as exc:
        print(f"[FAIL] environment reset failed: {exc}")
        return 1

    section_angle_length = _section_angle_length(initial_observation)
    if section_angle_length is None:
        print("[FAIL] observation missing robot_state or tip_pose.")
        return 2
    if section_angle_length != 6:
        print(f"[WARN] expected section angle length 6, got {section_angle_length}.")

    tip_source = _tip_source(initial_observation)
    initial = _snapshot(initial_observation)
    if initial is None:
        print("[FAIL] observation missing robot_state or tip_pose.")
        return 2

    results: dict[str, Any] = {}
    for name, base_command in _base_commands(args.amplitude).items():
        command = _fit_command(base_command, section_angle_length)
        result = _run_command(env, command, args.steps_per_command)
        if result is None:
            print("[FAIL] observation missing robot_state or tip_pose.")
            return 2
        results[name] = result
        print(
            f"[RESULT] command={name} "
            f"delta_tip_norm={result['delta_tip_norm']:.9f} "
            f"delta_qpos_norm={result['delta_qpos_norm']:.9f} "
            f"delta_ctrl_norm={result['delta_ctrl_norm']:.9f} "
            f"after_tip={result['after']['tip_position']}"
        )

    payload = {
        "amplitude": float(args.amplitude),
        "steps_per_command": int(args.steps_per_command),
        "section_angle_length": int(section_angle_length),
        "tip_source": tip_source,
        "initial": initial,
        "results": results,
    }
    if not _write_json(args.output, payload):
        return 3

    nonzero_results = [
        result
        for result in results.values()
        if _norm(result["command"]) > 1e-12
    ]
    if any(float(result["delta_tip_norm"]) > 1e-5 for result in nonzero_results):
        print("[OK] section_angles affect tip position.")
    elif any(
        float(result["delta_qpos_norm"]) > 1e-5 or float(result["delta_ctrl_norm"]) > 1e-5
        for result in nonzero_results
    ):
        print("[WARN] section_angles affect qpos/ctrl but tip position barely changes.")
    else:
        print("[FAIL] section_angles do not appear to affect qpos, ctrl, or tip position.")

    print(f"[OK] wrote {args.output}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
