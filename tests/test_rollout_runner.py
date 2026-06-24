from __future__ import annotations

from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.evaluation.rollout import RolloutConfig, run_rollout
from soft_continuum_vlm.policies.task_phase_expert_policy import TaskPhaseExpertPolicy
from soft_continuum_vlm.tasks.obstacle_avoid_pick_task import ObstacleAvoidPickTask


def test_run_rollout_returns_result_with_metrics_and_step_logs() -> None:
    config = RolloutConfig(task_name="obstacle_avoid_pick", env_type="mock", max_steps=12, seed=0)

    result = run_rollout(
        MockContinuumEnv(task="obstacle_avoid_pick", max_steps=12),
        TaskPhaseExpertPolicy(),
        ObstacleAvoidPickTask(),
        config,
    )

    assert result.task_name == "obstacle_avoid_pick"
    assert result.baseline == "task_phase_expert"
    assert result.seed == 0
    assert result.step_logs
    assert "max_contact_force" in result.metrics
