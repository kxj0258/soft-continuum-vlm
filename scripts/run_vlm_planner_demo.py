from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.controllers.vlm_planner_controller import VlmPlannerController
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic VLM planner demo.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--env", default="mock", choices=["mock"])
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.mock_env:
        raise ValueError("The first planner demo implementation supports --mock-env only.")
    env = MockContinuumEnv(task=args.task, max_steps=args.max_steps)
    controller = VlmPlannerController()
    obs = env.reset(task=args.task, language=args.language, seed=args.seed)
    step_logs: list[dict[str, object]] = []
    planner_output: dict[str, object] | None = None
    success = False
    try:
        for step_id in range(args.max_steps):
            action, info = controller.act(language=args.language, observation=obs, task_name=args.task)
            planner_output = info["planner_output"]
            obs, reward, done, env_info = env.step(action)
            success = success or bool(env_info.get("success", False))
            step_logs.append(
                {
                    "step_id": step_id,
                    "phase": info["phase"],
                    "reward": reward,
                    "done": done,
                    "action": action,
                    "contact": dict(obs["contact"]),
                    "info": info,
                }
            )
            if done:
                break
    finally:
        env.close()
    final_obs = {
        "robot_state": obs["robot_state"],
        "objects": obs["objects"],
        "contact": obs["contact"],
    }
    payload = {
        "task": args.task,
        "language": args.language,
        "planner_output": planner_output,
        "step_logs": step_logs,
        "success": success,
        "contact_metrics": {
            "max_contact_force": max((float(item["contact"].get("max_force", 0.0)) for item in step_logs), default=0.0),
            "max_penetration": max((float(item["contact"].get("max_penetration", 0.0)) for item in step_logs), default=0.0),
        },
        "final_observation_summary": final_obs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved VLM planner rollout to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
