from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.data.schema import ACTION_KEYS
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.evaluation.metrics import summarize_results
from soft_continuum_vlm.evaluation.rollout import RolloutConfig, run_rollout
from soft_continuum_vlm.policies.task_phase_expert_policy import TaskPhaseExpertPolicy
from soft_continuum_vlm.policies.vlm_planner_ik_policy import VlmPlannerIkPolicy
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
    parser = argparse.ArgumentParser(description="Evaluate policies on mock or real Feagine environments.")
    parser.add_argument("--tasks", nargs="+", required=True, choices=sorted(TASKS))
    parser.add_argument("--policies", nargs="+", required=True, choices=["task_phase_expert", "adapter", "vlm_planner_ik"])
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--env", default="mock", choices=["mock", "feagine_mujoco"])
    parser.add_argument("--config")
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--language", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_type = "mock" if args.mock_env or args.env == "mock" else "feagine_mujoco"
    episodes: list[dict[str, Any]] = []
    for task_name in args.tasks:
        for policy_name in args.policies:
            for episode_id in range(args.num_episodes):
                seed = args.seed + episode_id
                try:
                    env = make_env(args, env_type, task_name)
                    policy = make_policy(args, policy_name)
                    task = TASKS[task_name]()
                    result = run_rollout(
                        env,
                        policy,
                        task,
                        RolloutConfig(
                            task_name=task_name,
                            env_type=env_type,
                            max_steps=args.max_steps,
                            seed=seed,
                            language=args.language or task.language,
                        ),
                    )
                except Exception as exc:
                    if env_type == "feagine_mujoco":
                        print(f"Unable to run real Feagine policy evaluation: {exc}", file=sys.stderr)
                        return 2
                    raise
                row = {
                    "task": task_name,
                    "policy": policy_name,
                    "baseline": policy_name,
                    "seed": seed,
                    "success": bool(result.success),
                    "total_reward": result.total_reward,
                    "language": args.language or task.language,
                    **result.metrics,
                }
                episodes.append(row)
    payload = {
        "env_type": env_type,
        "metadata": {
            "action_schema": list(ACTION_KEYS),
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "policies": list(args.policies),
            "tasks": list(args.tasks),
        },
        "episodes": episodes,
        "summary": summarize_results(episodes),
    }
    write_outputs(payload, args.output, args.csv_output)
    print(f"Saved policy metrics to {args.output}")
    print(f"Saved policy CSV to {args.csv_output}")
    return 0


def make_env(args: argparse.Namespace, env_type: str, task_name: str) -> Any:
    if env_type == "mock":
        return MockContinuumEnv(task=task_name, max_steps=args.max_steps)
    config = load_yaml_config(args.config) if args.config else None
    return FeagineMujocoEnv(config)


def make_policy(args: argparse.Namespace, policy_name: str) -> Any:
    if policy_name == "task_phase_expert":
        return TaskPhaseExpertPolicy()
    if policy_name == "vlm_planner_ik":
        return VlmPlannerIkPolicy()
    if policy_name == "adapter":
        if not args.adapter_checkpoint:
            raise ValueError("--adapter-checkpoint is required for adapter policy.")
        from soft_continuum_vlm.policies.adapter_policy import AdapterPolicy

        return AdapterPolicy(args.adapter_checkpoint)
    raise ValueError(f"Unsupported policy: {policy_name}")


def write_outputs(payload: dict[str, Any], output: str, csv_output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in payload["episodes"] for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["episodes"])


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
