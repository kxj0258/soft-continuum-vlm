from __future__ import annotations

from soft_continuum_vlm.controllers.task_phase_expert import TaskPhaseExpert
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv


def test_task_phase_expert_returns_phase_target_safe_action_and_safety_info() -> None:
    env = MockContinuumEnv(task="pick_red_object", max_steps=20)
    observation = env.reset(seed=0)
    expert = TaskPhaseExpert()
    expert.reset("pick_red_object", observation["language"])

    action, info = expert.act(observation)

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert info["source"] == "task_phase_expert"
    assert info["task_name"] == "pick_red_object"
    assert info["phase"] == "approach_above_target"
    assert "target_state" in info
    assert "raw_action" in info
    assert "safe_action" in info
    assert "safety" in info
    assert "target_distance" in info


def test_task_phase_expert_advances_through_pick_phases() -> None:
    env = MockContinuumEnv(task="pick_red_object", max_steps=40)
    observation = env.reset(seed=0)
    expert = TaskPhaseExpert(phase_timeout=3)
    expert.reset("pick_red_object", observation["language"])

    phases = []
    transitions = []
    for _ in range(12):
        action, info = expert.act(observation)
        phases.append(info["phase"])
        transitions.append(info["phase_transition"])
        observation, _reward, done, _env_info = env.step(action)
        if done:
            break

    assert "close_gripper" in phases
    assert "lift" in phases or any(item["to"] == "lift" for item in transitions)
