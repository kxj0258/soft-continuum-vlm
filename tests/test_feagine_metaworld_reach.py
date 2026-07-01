from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest

from soft_continuum_vlm.controllers.reach_expert import FeagineReachExpert
from soft_continuum_vlm.envs.feagine_metaworld_env import FeagineMetaWorldEnv
from soft_continuum_vlm.tasks.feagine_reach_tasks import (
    FeagineReach3DTask,
    FeagineReachLeftTask,
    FeagineReachRightTask,
)


def _observation(tip=(0.0, 0.0, 0.3)) -> dict[str, Any]:
    return {
        "robot_state": {
            "tip_pose": {"position": list(tip)},
            "section_angles": [0.0] * 6,
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        "objects": {},
        "contact": {"max_force": 0.0, "max_penetration": 0.0, "contacts": []},
    }


def test_reach_task_uses_negative_distance_reward_and_threshold_success() -> None:
    task = FeagineReachLeftTask(goal_position=[-0.1, 0.0, 0.3], success_threshold=0.02)
    task.reset_task(seed=0, observation=_observation())

    far = _observation(tip=[0.0, 0.0, 0.3])
    near = _observation(tip=[-0.09, 0.0, 0.3])

    assert task.compute_reward(far) == pytest.approx(-0.1)
    assert task.compute_success(far) is False
    assert task.compute_success(near) is True
    assert task.get_goal() == [-0.1, 0.0, 0.3]
    assert task.get_task_info()["goal_mode"] == "fixed_world"


def test_reach_task_defaults_to_reset_tip_relative_goal() -> None:
    task = FeagineReachRightTask()
    task.reset_task(seed=0, observation=_observation(tip=[0.2, -0.1, 0.35]))

    np.testing.assert_allclose(task.get_goal(), [0.28, -0.1, 0.35])
    info = task.get_task_info()
    assert info["goal_offset"] == [0.08, 0.0, 0.0]
    assert info["goal_mode"] == "reset_tip_relative"


def test_reach_3d_goal_sampling_is_seeded_and_inside_bounds() -> None:
    first = FeagineReach3DTask()
    second = FeagineReach3DTask()

    first.reset_task(seed=11, observation=_observation())
    second.reset_task(seed=11, observation=_observation())

    assert first.get_goal() == second.get_goal()
    reset_tip = np.asarray(_observation()["robot_state"]["tip_pose"]["position"], dtype=np.float64)
    goal = np.asarray(first.get_goal()) - reset_tip
    low = np.asarray(first.goal_low)
    high = np.asarray(first.goal_high)
    assert np.all(goal >= low)
    assert np.all(goal <= high)
    center = (low + high) / 2.0
    radii = (high - low) / 2.0
    assert np.sum(((goal - center) / radii) ** 2) <= 1.0


def test_reach_expert_outputs_clipped_4d_action_toward_goal() -> None:
    observation = _observation(tip=[0.0, 0.0, 0.3])
    observation["task"] = {"goal_position": [0.02, -0.01, 0.3]}

    action = FeagineReachExpert(delta_xyz_scale=0.01).act(observation)

    np.testing.assert_allclose(action, [1.0, -1.0, 0.0, -1.0])


class _FakeLowLevelEnv:
    def __init__(self) -> None:
        self.observation = _observation()
        self.last_action: Mapping[str, Any] | None = None

    def reset(self, **_: Any) -> dict[str, Any]:
        return dict(self.observation)

    def step(self, action: Mapping[str, Any]):
        self.last_action = dict(action)
        return dict(self.observation), 123.0, False, {"backend": "fake"}

    def render(self):
        return None

    def close(self) -> None:
        return None

    def get_contact_info(self) -> Mapping[str, Any]:
        return self.observation["contact"]

    def get_robot_state(self) -> Mapping[str, Any]:
        return self.observation["robot_state"]


def test_metaworld_wrapper_exposes_4d_action_and_runtime_conversion() -> None:
    backend = _FakeLowLevelEnv()
    task = FeagineReachRightTask(goal_position=[0.0, 0.0, 0.3])
    env = FeagineMetaWorldEnv(backend, task)

    observation = env.reset(seed=3)
    next_observation, reward, done, info = env.step([0.0, 0.0, 0.0, -1.0])

    assert env.action_space.shape == (4,)
    assert observation["task"]["name"] == "feagine_reach_right"
    assert next_observation["task"]["goal_position"] == [0.0, 0.0, 0.3]
    assert set(backend.last_action or {}) == {
        "section_angles",
        "grip_command",
        "grasper_rotation",
    }
    assert reward == pytest.approx(0.0)
    assert done is True
    assert info["success"] is True
    assert info["ik_success_rate"] == pytest.approx(1.0)
    assert info["tip_goal_distance_history"] == [0.0]
