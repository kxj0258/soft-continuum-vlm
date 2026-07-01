from __future__ import annotations

import numpy as np
import pytest

from soft_continuum_vlm.controllers.push_expert import FeaginePushExpert
from soft_continuum_vlm.tasks.feagine_push_tasks import (
    FeagineContactPushTask,
    FeaginePushLeftToRightTask,
    FeaginePushRightToLeftTask,
)


def _observation(
    *,
    tip=(0.0, 0.0, 0.2),
    object_position=(-0.04, 0.0, 0.2),
    max_force=0.0,
    max_penetration=0.0,
    contacts=None,
):
    return {
        "robot_state": {
            "tip_pose": {"position": list(tip)},
            "section_angles": [0.0] * 6,
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        "objects": {
            "push_object": {
                "pose": {
                    "position": list(object_position),
                    "orientation": [1.0, 0.0, 0.0, 0.0],
                }
            }
        },
        "contact": {
            "max_force": max_force,
            "max_penetration": max_penetration,
            "contacts": list(contacts or []),
        },
    }


def test_push_reward_combines_distances_contact_penalty_and_success_bonus() -> None:
    task = FeaginePushLeftToRightTask(
        goal_position=[0.08, 0.0, 0.2],
        success_threshold=0.02,
        success_bonus=2.0,
    )
    observation = _observation(
        tip=[0.07, 0.0, 0.2],
        object_position=[0.08, 0.0, 0.2],
        max_force=0.5,
        contacts=[{"geom1": "finger_left", "geom2": "push_object"}],
    )
    task.reset_task(observation=observation)
    evaluation = task.evaluate(observation)

    assert evaluation.success is True
    assert evaluation.reward > 1.0
    metrics = evaluation.metrics
    assert metrics["object_goal_distance"] == pytest.approx(0.0)
    assert metrics["max_contact_force"] == pytest.approx(0.5)


def test_push_metrics_distinguish_target_contact_from_wrong_collision() -> None:
    task = FeagineContactPushTask()
    initial = _observation(object_position=[0.0, 0.0, 0.2])
    task.reset_task(observation=initial)
    observation = _observation(
        object_position=[0.0, 0.02, 0.2],
        contacts=[
            {"geom1": "finger_left", "geom2": "push_object"},
            {"geom1": "arm_section", "geom2": "shelf"},
        ]
    )

    metrics = task.compute_metrics(observation)

    assert metrics["target_contact_flag"] == 1.0
    assert metrics["wrong_contact_count"] == 1.0
    assert metrics["object_displacement"] == pytest.approx(0.02)


def test_push_task_transitions_from_approach_to_push() -> None:
    task = FeaginePushRightToLeftTask(approach_threshold=0.03)
    task.reset_task(observation=_observation(tip=[0.1, 0.0, 0.2], object_position=[0.04, 0.0, 0.2]))

    task.evaluate(_observation(tip=[0.05, 0.0, 0.2], object_position=[0.04, 0.0, 0.2]))

    assert task.get_task_info()["phase"] == "push"


def test_push_expert_uses_open_gripper_and_pushes_toward_goal() -> None:
    observation = _observation(tip=[-0.04, 0.0, 0.2], object_position=[-0.04, 0.0, 0.2])
    observation["task"] = {
        "phase": "push",
        "object_name": "push_object",
        "goal_position": [0.08, 0.0, 0.2],
    }

    action = FeaginePushExpert(delta_xyz_scale=0.01).act(observation)

    assert action.shape == (4,)
    assert action[0] == pytest.approx(1.0)
    assert action[1] == pytest.approx(0.0)
    assert action[2] == pytest.approx(0.0)
    assert action[3] == pytest.approx(-1.0)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
