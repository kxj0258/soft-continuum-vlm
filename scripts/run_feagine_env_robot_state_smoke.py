from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test FeagineMujocoEnv robot_state observation.")
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(np.asarray(value).shape)


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64).reshape(-1)))


def _contact_summary(contact: Mapping[str, Any]) -> str:
    contacts = contact.get("contacts", []) or []
    return (
        f"contacts={len(contacts)}, "
        f"max_force={float(contact.get('max_force', 0.0)):.6f}, "
        f"max_penetration={float(contact.get('max_penetration', 0.0)):.6f}"
    )


def main() -> int:
    args = _parse_args()
    try:
        from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv
    except Exception as exc:
        print(f"[FAIL] could not import FeagineMujocoEnv: {exc}")
        return 1

    env = FeagineMujocoEnv(
        {
            "env": {
                "robot_preset": "a03_type_2",
                "asset_model_type": "mjcf",
                "render_mode": "none",
                "max_episode_steps": max(args.steps + 1, 2),
            }
        }
    )

    try:
        observation = env.reset()
    except Exception as exc:
        print(f"[FAIL] reset failed: {exc}")
        return 1

    robot_state = observation.get("robot_state", {})
    tip_pose = robot_state.get("tip_pose", {})
    section_angles = list(robot_state.get("section_angles", []) or [])
    section_angle_length = len(section_angles)
    if section_angle_length == 0:
        print("[WARN] section_angles length is 0; zero action will use an empty section_angles list.")

    print(f"[INFO] observation keys: {sorted(observation.keys())}")
    print(f"[INFO] robot_state keys: {sorted(robot_state.keys())}")
    print(f"[INFO] qpos shape: {_shape(robot_state.get('qpos', []))}")
    print(f"[INFO] qvel shape: {_shape(robot_state.get('qvel', []))}")
    print(f"[INFO] ctrl shape: {_shape(robot_state.get('ctrl', []))}")
    print(f"[INFO] tip_pose source: {tip_pose.get('source')}")
    print(f"[INFO] tip_pose position: {tip_pose.get('position')}")
    print(f"[INFO] section_angles length: {section_angle_length}")
    print(f"[INFO] grip_command: {robot_state.get('grip_command')}")
    print(f"[INFO] grasper_rotation: {robot_state.get('grasper_rotation')}")

    action = {
        "section_angles": [0.0] * section_angle_length,
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    }
    for step_index in range(1, args.steps + 1):
        try:
            observation, _reward, _done, _info = env.step(action)
        except Exception as exc:
            print(f"[FAIL] step {step_index} failed: {exc}")
            return 1
        robot_state = observation.get("robot_state", {})
        tip_pose = robot_state.get("tip_pose", {})
        print(
            f"[INFO] step={step_index}, "
            f"tip_position={tip_pose.get('position')}, "
            f"qpos_norm={_norm(robot_state.get('qpos', [])):.6f}, "
            f"qvel_norm={_norm(robot_state.get('qvel', [])):.6f}, "
            f"ctrl_norm={_norm(robot_state.get('ctrl', [])):.6f}, "
            f"contact=({_contact_summary(observation.get('contact', {}))})"
        )

    if robot_state.get("tip_pose", {}).get("source") == "unresolved":
        print("[WARN] tip pose unresolved; expected body name is feagine_grasper_tip.")

    env.close()
    print(f"[INFO] closed: {env.get_robot_state().get('closed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
