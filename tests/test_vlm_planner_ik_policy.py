from __future__ import annotations

from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.policies.vlm_planner_ik_policy import VlmPlannerIkPolicy


def test_vlm_planner_ik_policy_logs_planner_output_and_safe_action() -> None:
    observation = MockContinuumEnv(task="obstacle_avoid_pick").reset(
        language="avoid the obstacle and gently pick the red object",
        seed=0,
    )
    policy = VlmPlannerIkPolicy()
    policy.reset("obstacle_avoid_pick", observation["language"])

    action, info = policy.act(observation)

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert info["source"] == "vlm_planner_ik_policy"
    assert "planner_output" in info
    assert "selected_subgoal" in info
    assert "phase" in info
    assert "safe_action" in info
