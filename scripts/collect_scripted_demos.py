from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController
from soft_continuum_vlm.data.features import build_morphology_vector, encode_language_stub
from soft_continuum_vlm.data.schema import (
    ACTION_KEYS,
    OBSERVATION_SCHEMA_VERSION,
    flatten_action,
    flatten_contact,
    flatten_proprioception,
)
from soft_continuum_vlm.data.serialization import save_demo_npz
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted demonstrations.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--env", default="mock", choices=["mock", "feagine_mujoco"])
    parser.add_argument("--config")
    parser.add_argument("--save-config-snapshot", action="store_true")
    return parser.parse_args()


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def make_env(args: argparse.Namespace):
    if args.mock_env or args.env == "mock":
        return MockContinuumEnv(task=args.task, max_steps=args.max_steps)
    config = load_yaml_config(args.config) if args.config else None
    return FeagineMujocoEnv(config)


def scripted_action(controller: PccIkController, task: str, step_id: int) -> dict[str, Any]:
    action = controller.compute_action(target_state={}, robot_state={})
    if task in {"pick_red_object", "obstacle_avoid_pick"}:
        action["section_angles"] = [0.2] * 6
        action["grip_command"] = 1.0 if step_id >= 2 else 0.0
    elif task == "contact_push":
        action["section_angles"] = [0.15] * 6
        action["grip_command"] = 0.0
    elif task == "rotate_and_place":
        action["section_angles"] = [0.0] * 6
        action["grip_command"] = 1.0
        action["grasper_rotation"] = 1.0
    return action


def append_step(rows: dict[str, list[Any]], *, obs: dict[str, Any], action: dict[str, Any],
                reward: float, done: bool, success: bool, task_name: str,
                episode_id: int, step_id: int, phase: str, morphology: np.ndarray) -> None:
    rows["proprioception"].append(flatten_proprioception(obs))
    rows["contact"].append(flatten_contact(obs["contact"]))
    rows["language"].append(str(obs["language"]))
    rows["language_feature"].append(encode_language_stub(str(obs["language"])))
    rows["morphology"].append(morphology)
    rows["action"].append(json.dumps(action, sort_keys=True))
    rows["action_vector"].append(flatten_action(action, section_count=3))
    rows["reward"].append(float(reward))
    rows["done"].append(bool(done))
    rows["success"].append(bool(success))
    rows["task_name"].append(task_name)
    rows["phase"].append(phase)
    rows["episode_id"].append(int(episode_id))
    rows["step_id"].append(int(step_id))


def rows_to_arrays(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {
        "proprioception": np.asarray(rows["proprioception"], dtype=np.float32),
        "contact": np.asarray(rows["contact"], dtype=np.float32),
        "language": np.asarray(rows["language"]),
        "language_feature": np.asarray(rows["language_feature"], dtype=np.float32),
        "morphology": np.asarray(rows["morphology"], dtype=np.float32),
        "action": np.asarray(rows["action"]),
        "action_vector": np.asarray(rows["action_vector"], dtype=np.float32),
        "reward": np.asarray(rows["reward"], dtype=np.float32),
        "done": np.asarray(rows["done"], dtype=bool),
        "success": np.asarray(rows["success"], dtype=bool),
        "task_name": np.asarray(rows["task_name"]),
        "phase": np.asarray(rows["phase"]),
        "episode_id": np.asarray(rows["episode_id"], dtype=np.int32),
        "step_id": np.asarray(rows["step_id"], dtype=np.int32),
    }


def main() -> int:
    args = parse_args()
    env = make_env(args)
    controller = PccIkController(section_count=3)
    morphology = build_morphology_vector(section_count=3)
    rows: dict[str, list[Any]] = {key: [] for key in (
        "proprioception", "contact", "language", "language_feature", "morphology",
        "action", "action_vector", "reward", "done", "success", "task_name",
        "phase", "episode_id", "step_id"
    )}
    try:
        for episode_id in range(args.num_episodes):
            obs = env.reset(task=args.task, seed=args.seed + episode_id)
            for step_id in range(args.max_steps):
                action = scripted_action(controller, args.task, step_id)
                next_obs, reward, done, info = env.step(action)
                append_step(
                    rows,
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    success=bool(info.get("success", False)),
                    task_name=args.task,
                    episode_id=episode_id,
                    step_id=step_id,
                    phase="scripted",
                    morphology=morphology,
                )
                obs = next_obs
                if done:
                    break
    finally:
        env.close()

    output = Path(args.output)
    arrays = rows_to_arrays(rows)
    save_demo_npz(output, **arrays)
    metadata = {
        "git_commit": git_commit_hash(),
        "command_line": sys.argv,
        "seed": args.seed,
        "task_config": {"task": args.task},
        "method_config": {"method": "scripted_pcc_ik"},
        "env_type": "mock" if args.mock_env or args.env == "mock" else args.env,
        "action_schema": list(ACTION_KEYS),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "num_steps": int(arrays["action_vector"].shape[0]),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.save_config_snapshot:
        output.with_suffix(".config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {metadata['num_steps']} scripted transitions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
