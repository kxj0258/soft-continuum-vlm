from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.calibrated_jacobian import JacobianCalibration
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate local section_angles to tip_position Jacobian.")
    parser.add_argument("--preset", default="a03_type_2")
    parser.add_argument("--model-type", default="mjcf")
    parser.add_argument("--perturbation", type=float, default=0.05)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--active-threshold", type=float, default=1e-5)
    parser.add_argument("--output", default="outputs/calibration/a03_type_2_section_jacobian.json")
    return parser.parse_args()


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _norm(value: Any) -> float:
    return float(np.linalg.norm(_as_array(value)))


def _robot_state(observation: dict[str, Any]) -> dict[str, Any] | None:
    robot_state = observation.get("robot_state")
    if not isinstance(robot_state, dict):
        return None
    tip_pose = robot_state.get("tip_pose")
    if not isinstance(tip_pose, dict):
        return None
    return robot_state


def _tip_position(robot_state: dict[str, Any]) -> np.ndarray:
    return _as_array(robot_state["tip_pose"].get("position", [0.0, 0.0, 0.0]))[:3]


def _tip_source(robot_state: dict[str, Any]) -> str:
    return str(robot_state["tip_pose"].get("source", "unresolved"))


def _snapshot(observation: dict[str, Any]) -> dict[str, np.ndarray] | None:
    robot_state = _robot_state(observation)
    if robot_state is None:
        return None
    return {
        "tip": _tip_position(robot_state),
        "qpos": _as_array(robot_state.get("qpos", [])),
        "ctrl": _as_array(robot_state.get("ctrl", [])),
        "section_angles": _as_array(robot_state.get("section_angles", [])),
    }


def _make_env(preset: str, model_type: str, settle_steps: int) -> FeagineMujocoEnv:
    return FeagineMujocoEnv(
        {
            "env": {
                "robot_preset": preset,
                "asset_model_type": model_type,
                "render_mode": "none",
                "max_episode_steps": max(int(settle_steps) + 1, 2),
            }
        }
    )


def _run_perturbation(
    env: FeagineMujocoEnv,
    section_angles: list[float],
    settle_steps: int,
) -> dict[str, np.ndarray] | None:
    env.reset()
    action = {
        "section_angles": section_angles,
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    }
    observation: dict[str, Any] | None = None
    for _ in range(settle_steps):
        observation, _reward, _done, _info = env.step(action)
    if observation is None:
        observation = env.get_observation()
    return _snapshot(observation)


def _condition_number(jacobian: np.ndarray, active_columns: list[int]) -> float | None:
    if len(active_columns) < 1:
        return None
    active = jacobian[:, active_columns]
    if active.ndim != 2 or active.size == 0:
        return None
    try:
        value = float(np.linalg.cond(active))
    except Exception:
        return None
    return value if np.isfinite(value) else None


def main() -> int:
    args = _parse_args()
    try:
        env = _make_env(args.preset, args.model_type, args.settle_steps)
    except Exception as exc:
        print(f"[FAIL] environment creation failed: {exc}")
        return 1
    print("[OK] created FeagineMujocoEnv")

    try:
        observation = env.reset()
    except Exception as exc:
        print(f"[FAIL] environment reset failed: {exc}")
        return 1

    robot_state = _robot_state(observation)
    if robot_state is None:
        print("[FAIL] observation missing robot_state or tip_pose.")
        return 2

    tip_source = _tip_source(robot_state)
    print(f"[INFO] tip_source={tip_source}")
    if tip_source != "body:feagine_grasper_tip":
        print(f"[WARN] unexpected tip_source={tip_source}; expected body:feagine_grasper_tip.")
        if tip_source == "unresolved":
            return 3

    base_snapshot = _snapshot(observation)
    if base_snapshot is None:
        print("[FAIL] observation missing robot_state or tip_pose.")
        return 2
    base_section_angles = base_snapshot["section_angles"].tolist()
    base_tip = base_snapshot["tip"]
    section_angle_length = len(base_section_angles)
    print(f"[INFO] section_angle_length={section_angle_length}")
    print(f"[INFO] base_tip={base_tip.tolist()}")
    if section_angle_length <= 0:
        return 4

    columns: list[np.ndarray] = []
    per_dimension: list[dict[str, Any]] = []
    perturbation = float(args.perturbation)
    threshold = float(args.active_threshold)

    for dim in range(section_angle_length):
        plus_command = list(base_section_angles)
        plus_command[dim] += perturbation
        plus_snapshot = _run_perturbation(env, plus_command, args.settle_steps)
        if plus_snapshot is None:
            print("[FAIL] observation missing robot_state or tip_pose.")
            return 2

        minus_command = list(base_section_angles)
        minus_command[dim] -= perturbation
        minus_snapshot = _run_perturbation(env, minus_command, args.settle_steps)
        if minus_snapshot is None:
            print("[FAIL] observation missing robot_state or tip_pose.")
            return 2

        tip_plus = plus_snapshot["tip"]
        tip_minus = minus_snapshot["tip"]
        delta_tip_plus = tip_plus - base_tip
        delta_tip_minus = tip_minus - base_tip
        delta_tip_plus_norm = _norm(delta_tip_plus)
        delta_tip_minus_norm = _norm(delta_tip_minus)

        plus_active = delta_tip_plus_norm > threshold
        minus_active = delta_tip_minus_norm > threshold
        if plus_active and minus_active:
            column = (tip_plus - tip_minus) / (2.0 * perturbation)
            method = "central"
        elif plus_active:
            column = (tip_plus - base_tip) / perturbation
            method = "forward_only"
        elif minus_active:
            column = (base_tip - tip_minus) / perturbation
            method = "backward_only"
        else:
            column = np.zeros(3, dtype=np.float64)
            method = "inactive"

        column_norm = _norm(column)
        columns.append(column)
        per_dimension.append(
            {
                "dim": dim,
                "plus": {
                    "delta_tip": delta_tip_plus.tolist(),
                    "delta_tip_norm": delta_tip_plus_norm,
                    "delta_qpos_norm": _norm(plus_snapshot["qpos"] - base_snapshot["qpos"]),
                    "delta_ctrl_norm": _norm(plus_snapshot["ctrl"] - base_snapshot["ctrl"]),
                },
                "minus": {
                    "delta_tip": delta_tip_minus.tolist(),
                    "delta_tip_norm": delta_tip_minus_norm,
                    "delta_qpos_norm": _norm(minus_snapshot["qpos"] - base_snapshot["qpos"]),
                    "delta_ctrl_norm": _norm(minus_snapshot["ctrl"] - base_snapshot["ctrl"]),
                },
                "method": method,
                "column": column.tolist(),
            }
        )
        print(
            f"[DIM {dim}] plus_tip_norm={delta_tip_plus_norm:.9f} "
            f"minus_tip_norm={delta_tip_minus_norm:.9f} "
            f"method={method} column_norm={column_norm:.9f}"
        )

    jacobian = np.column_stack(columns)
    column_norms = np.linalg.norm(jacobian, axis=0)
    active_columns = [int(index) for index, value in enumerate(column_norms) if float(value) > threshold]
    inactive_columns = [int(index) for index, value in enumerate(column_norms) if float(value) <= threshold]
    rank = int(np.linalg.matrix_rank(jacobian, tol=threshold))
    condition_number = _condition_number(jacobian, active_columns)

    calibration = JacobianCalibration(
        preset=args.preset,
        model_type=args.model_type,
        section_angle_length=section_angle_length,
        base_section_angles=[float(value) for value in base_section_angles],
        base_tip_position=[float(value) for value in base_tip.tolist()],
        jacobian=jacobian.tolist(),
        perturbation=perturbation,
        settle_steps=int(args.settle_steps),
        tip_source=tip_source,
        active_columns=active_columns,
        inactive_columns=inactive_columns,
        column_norms=[float(value) for value in column_norms.tolist()],
        rank=rank,
        condition_number=condition_number,
        metadata={
            "per_dimension": per_dimension,
            "active_threshold": threshold,
            "notes": [
                "This calibration is local around the reset configuration.",
                "Some dimensions may be inactive due to Feagine section angle mapping or command saturation.",
                "Use this JSON only for local Jacobian IK debugging.",
            ],
        },
    )

    print(f"[RESULT] J shape={calibration.as_array().shape}")
    print(f"[RESULT] rank={rank}")
    print(f"[RESULT] active_columns={active_columns}")
    print(f"[RESULT] inactive_columns={inactive_columns}")
    print(f"[RESULT] column_norms={calibration.column_norms}")
    if not active_columns:
        print("[WARN] all Jacobian columns are inactive; cannot use this for IK yet.")

    try:
        calibration.save_json(args.output)
    except Exception as exc:
        print(f"[FAIL] JSON save failed: {exc}")
        return 5
    print(f"[OK] wrote {args.output}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
