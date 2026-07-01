from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pytest

from soft_continuum_vlm.envs.feagine_gym_state_env import FeagineGymStateEnv
from soft_continuum_vlm.tasks import make_feagine_metaworld_task
from soft_continuum_vlm.tasks.feagine_reach_tasks import FeagineReachRightTask


def _observation(
    *,
    tip=(0.0, 0.0, 0.3),
    object_name: str = "pick_object",
    object_position=(0.0, 0.0, 0.2),
    grasped: bool = False,
) -> dict[str, Any]:
    return {
        "robot_state": {
            "tip_pose": {"position": list(tip)},
            "section_angles": [0.0] * 6,
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        "objects": {
            object_name: {
                "pose": {
                    "position": list(object_position),
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                },
                "grasped": grasped,
                "principal_axis": [1.0, 0.0, 0.0],
            }
        },
        "contact": {"max_force": 0.0, "max_penetration": 0.0, "contacts": []},
    }


class _FakeBackend:
    def __init__(self, observation: Mapping[str, Any]) -> None:
        self.observation = dict(observation)
        self.last_action: Mapping[str, Any] | None = None

    def reset(self, **_: Any) -> dict[str, Any]:
        return dict(self.observation)

    def step(self, action: Mapping[str, Any]):
        self.last_action = dict(action)
        return dict(self.observation), 0.0, False, {"backend": "fake"}

    def render(self):
        return None

    def close(self) -> None:
        return None


def test_task_factory_resolves_reach_push_and_pick_place_names() -> None:
    assert make_feagine_metaworld_task("feagine_reach_left").name == "feagine_reach_left"
    assert make_feagine_metaworld_task("feagine_contact_push").name == "feagine_contact_push"
    assert (
        make_feagine_metaworld_task("feagine_pick_left_place_right").name
        == "feagine_pick_left_place_right"
    )


def test_gym_state_env_reset_returns_goal_conditioned_state_and_info() -> None:
    backend = _FakeBackend(_observation(tip=[0.0, 0.0, 0.3]))
    env = FeagineGymStateEnv(backend, make_feagine_metaworld_task("feagine_reach_right"))

    observation, info = env.reset(seed=5)

    assert set(observation) == {"observation", "achieved_goal", "desired_goal"}
    assert observation["observation"].dtype == np.float32
    np.testing.assert_allclose(observation["achieved_goal"], [0.0, 0.0, 0.3])
    np.testing.assert_allclose(observation["desired_goal"], [0.08, 0.0, 0.3])
    assert info["task_name"] == "feagine_reach_right"
    assert info["success"] is False
    assert info["task_metrics"]["tip_goal_distance"] > 0.0
    assert env.get_raw_observation()["task"]["name"] == "feagine_reach_right"


def test_gym_state_env_step_returns_terminated_and_truncated_separately() -> None:
    backend = _FakeBackend(_observation(tip=[0.08, 0.0, 0.24]))
    env = FeagineGymStateEnv(
        backend,
        FeagineReachRightTask(goal_position=[0.08, 0.0, 0.24]),
        max_episode_steps=3,
    )
    env.reset(seed=0)

    observation, reward, terminated, truncated, info = env.step([0.0, 0.0, 0.0, -1.0])

    assert observation["observation"].shape[0] >= 12
    assert reward == pytest.approx(0.0)
    assert terminated is True
    assert truncated is False
    assert info["step_count"] == 1
    assert info["success"] is True
    assert set(backend.last_action or {}) == {
        "section_angles",
        "grip_command",
        "grasper_rotation",
    }


def test_gym_state_env_truncates_at_max_episode_steps_without_success() -> None:
    backend = _FakeBackend(_observation(tip=[0.0, 0.0, 0.3]))
    env = FeagineGymStateEnv(
        backend,
        make_feagine_metaworld_task("feagine_reach_right"),
        max_episode_steps=1,
    )
    env.reset(seed=0)

    next_observation, _reward, terminated, truncated, info = env.step([0.0, 0.0, 0.0, -1.0])

    assert set(next_observation) == {"observation", "achieved_goal", "desired_goal"}
    assert terminated is False
    assert truncated is True
    assert info["backend_done"] is False
