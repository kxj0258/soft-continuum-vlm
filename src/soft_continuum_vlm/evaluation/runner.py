from __future__ import annotations

from typing import Any

import numpy as np

from soft_continuum_vlm.controllers.pcc_ik_controller import PccIkController
from soft_continuum_vlm.controllers.scripted_expert import ScriptedExpert
from soft_continuum_vlm.controllers.vlm_planner_controller import VlmPlannerController
from soft_continuum_vlm.data.schema import unflatten_action
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.evaluation.metrics import summarize_episode_logs, summarize_results


def run_baseline_episode(
    *,
    task: str,
    baseline: str,
    seed: int,
    max_steps: int,
    language: str | None = None,
) -> dict[str, Any]:
    env = MockContinuumEnv(task=task, max_steps=max_steps)
    obs = env.reset(task=task, seed=seed, language=language)
    step_logs: list[dict[str, Any]] = []
    scripted = ScriptedExpert(controller=PccIkController())
    vlm = VlmPlannerController()
    rng = np.random.default_rng(seed)
    episode_success = False
    try:
        for step_id in range(max_steps):
            if baseline == "scripted_expert":
                action, info = scripted.act(obs)
            elif baseline == "vlm_planner_ik":
                action, info = vlm.act(
                    language=str(language or obs.get("language", "")),
                    observation=obs,
                    task_name=task,
                )
            elif baseline == "adapter":
                action = unflatten_action(rng.normal(0.0, 0.05, size=8).astype(np.float32), section_count=3)
                info = {"warning": "adapter baseline uses random initialized policy when no checkpoint is provided"}
            else:
                raise ValueError(f"Unsupported baseline: {baseline}")
            obs, reward, done, env_info = env.step(action)
            episode_success = episode_success or bool(env_info.get("success", False))
            step_logs.append(
                {
                    "step_id": step_id,
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
    metrics = summarize_episode_logs(step_logs, success=episode_success)
    metrics.update(
        {
            "task": task,
            "baseline": baseline,
            "seed": seed,
            "language": str(language or ""),
        }
    )
    return metrics


def run_baseline_suite(
    *,
    tasks: list[str],
    baselines: list[str],
    num_episodes: int,
    max_steps: int,
    language: str = "",
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for task in tasks:
        for baseline in baselines:
            for episode_id in range(num_episodes):
                episodes.append(
                    run_baseline_episode(
                        task=task,
                        baseline=baseline,
                        seed=episode_id,
                        max_steps=max_steps,
                        language=language,
                    )
                )
    return {
        "env_type": "mock",
        "episodes": episodes,
        "summary": summarize_results(episodes),
    }
