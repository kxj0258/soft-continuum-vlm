from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.task_phase_expert import TaskPhaseExpert
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.evaluation.metrics import compute_rollout_metrics
from soft_continuum_vlm.tasks.contact_push_task import ContactPushTask
from soft_continuum_vlm.tasks.obstacle_avoid_pick_task import ObstacleAvoidPickTask
from soft_continuum_vlm.tasks.pick_task import PickTask
from soft_continuum_vlm.tasks.rotate_place_task import RotatePlaceTask
from soft_continuum_vlm.utils.config import load_yaml_config


TASKS = {
    "pick_red_object": PickTask,
    "obstacle_avoid_pick": ObstacleAvoidPickTask,
    "contact_push": ContactPushTask,
    "rotate_and_place": RotatePlaceTask,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the task-phase scripted expert.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--env", default="mock", choices=["mock", "feagine_mujoco"])
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_type = "mock" if args.mock_env or args.env == "mock" else "feagine_mujoco"
    try:
        env = make_env(args, env_type)
        observation = env.reset(task=args.task, seed=args.seed)
    except Exception as exc:
        print(f"Unable to initialize {env_type} environment for task-phase expert: {exc}", file=sys.stderr)
        return 2

    task = TASKS[args.task]()
    expert = TaskPhaseExpert()
    expert.reset(args.task, str(observation.get("language", task.language)))
    step_logs: list[dict[str, Any]] = []
    episode_success = False
    try:
        for step_id in range(args.max_steps):
            action, expert_info = expert.act(observation)
            next_observation, reward, done, env_info = env.step(action)
            task_result = task.evaluate(next_observation)
            success = bool(env_info.get("success", task_result.get("success", False)))
            episode_success = episode_success or success
            step_log = {
                "step_id": step_id,
                "observation_summary": summarize_observation(observation),
                "phase": expert_info["phase"],
                "target_state": expert_info["target_state"],
                "action": action,
                "raw_action": expert_info["raw_action"],
                "safe_action": expert_info["safe_action"],
                "safety": expert_info["safety"],
                "contact": dict(next_observation.get("contact", {})),
                "reward": float(reward),
                "done": bool(done),
                "success": success,
                "metrics": dict(task_result.get("metrics", env_info.get("metrics", {}))),
                "target_distance": float(expert_info.get("target_distance", 0.0)),
            }
            step_logs.append(step_log)
            observation = next_observation
            if done:
                break
    finally:
        env.close()

    metrics = compute_rollout_metrics(step_logs)
    metrics["success"] = 1.0 if episode_success else metrics["success"]
    payload = {
        "task": args.task,
        "env_type": env_type,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "expert": {
            "source": "task_phase_expert",
            "controller": "PccIkController",
            "safety_mode": "hold_current",
        },
        "steps": step_logs,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved task-phase expert rollout to {output}")
    return 0


def make_env(args: argparse.Namespace, env_type: str) -> Any:
    if env_type == "mock":
        return MockContinuumEnv(task=args.task, max_steps=args.max_steps)
    config = load_yaml_config(args.config) if args.config else None
    return FeagineMujocoEnv(config)


def summarize_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    robot_state = observation.get("robot_state", {})
    contact = observation.get("contact", {})
    objects = observation.get("objects", {})
    return {
        "robot_state_keys": sorted(robot_state.keys()) if isinstance(robot_state, Mapping) else [],
        "object_availability": {
            str(key): bool(value.get("available", True)) if isinstance(value, Mapping) else False
            for key, value in objects.items()
        }
        if isinstance(objects, Mapping)
        else {},
        "contact_summary": {
            "max_force": float(contact.get("max_force", 0.0)),
            "max_penetration": float(contact.get("max_penetration", 0.0)),
            "contact_count": len(contact.get("contacts", [])),
        }
        if isinstance(contact, Mapping)
        else {},
    }


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
