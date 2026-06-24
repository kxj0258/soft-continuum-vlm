from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from soft_continuum_vlm.evaluation.metrics import compute_rollout_metrics


@dataclass
class RolloutConfig:
    task_name: str
    env_type: str
    max_steps: int
    seed: int
    language: str | None = None


@dataclass
class RolloutResult:
    task_name: str
    baseline: str
    seed: int
    success: bool
    total_reward: float
    metrics: dict[str, float]
    step_logs: list[dict[str, Any]]


def run_rollout(env: Any, policy: Any, task: Any, config: RolloutConfig) -> RolloutResult:
    observation = env.reset(task=config.task_name, seed=config.seed, language=config.language)
    language = str(config.language or observation.get("language", getattr(task, "language", "")))
    policy.reset(config.task_name, language)
    step_logs: list[dict[str, Any]] = []
    success = False
    try:
        for step_id in range(config.max_steps):
            action, policy_info = policy.act(observation)
            next_observation, reward, done, env_info = env.step(action)
            task_result = task.evaluate(next_observation)
            step_success = bool(env_info.get("success", task_result.get("success", False)))
            success = success or step_success
            step_logs.append(
                {
                    "step_id": step_id,
                    "phase": policy_info.get("phase", ""),
                    "target_state": policy_info.get("target_state", {}),
                    "target_distance": float(policy_info.get("target_distance", 0.0)),
                    "action": action,
                    "raw_action": policy_info.get("raw_action", action),
                    "safe_action": policy_info.get("safe_action", action),
                    "safety": policy_info.get("safety", policy_info.get("safety_info", {})),
                    "contact": dict(next_observation.get("contact", {})),
                    "reward": float(reward),
                    "done": bool(done),
                    "success": step_success,
                    "metrics": dict(task_result.get("metrics", env_info.get("metrics", {}))),
                    "info": dict(policy_info),
                }
            )
            observation = next_observation
            if done:
                break
    finally:
        env.close()
    metrics = compute_rollout_metrics(step_logs)
    total_reward = float(metrics["total_reward"])
    return RolloutResult(
        task_name=config.task_name,
        baseline=str(getattr(policy, "baseline_name", policy.__class__.__name__)),
        seed=config.seed,
        success=success or bool(metrics["success"]),
        total_reward=total_reward,
        metrics=metrics,
        step_logs=step_logs,
    )
