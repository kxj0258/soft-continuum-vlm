from __future__ import annotations

import pytest

from soft_continuum_vlm.tasks.contact_push_task import ContactPushTask
from soft_continuum_vlm.tasks.obstacle_avoid_pick_task import ObstacleAvoidPickTask
from soft_continuum_vlm.tasks.pick_task import PickTask
from soft_continuum_vlm.tasks.rotate_place_task import RotatePlaceTask


def base_observation() -> dict[str, object]:
    return {
        "robot_state": {
            "tip_pose": {
                "position": [0.5, 0.0, 0.2],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
            "section_angles": [0.0] * 6,
            "grip_command": 1.0,
            "grasper_rotation": 0.0,
        },
        "objects": {},
        "contact": {
            "max_force": 0.0,
            "max_penetration": 0.0,
            "contacts": [],
        },
    }


def test_pick_task_requires_grasped_and_lifted_target() -> None:
    observation = base_observation()
    observation["objects"] = {
        "red_object": {
            "pose": {
                "position": [0.5, 0.0, 0.16],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
            "grasped": True,
        }
    }

    result = PickTask().evaluate(observation)

    assert result["success"] is True
    assert result["metrics"]["target_grasped"] is True
    assert result["metrics"]["lift_height"] == pytest.approx(0.16)


def test_obstacle_avoid_pick_fails_when_obstacle_contact_exceeds_limit() -> None:
    observation = base_observation()
    observation["objects"] = {"target_object": {"pose": {"position": [0.5, 0.0, 0.2]}, "grasped": True}}
    observation["contact"] = {
        "max_force": 0.9,
        "max_penetration": 0.0,
        "contacts": [
            {
                "geom1": "arm_section_2",
                "geom2": "obstacle",
                "position": [0.4, 0.0, 0.1],
                "normal": [0.0, 1.0, 0.0],
                "force": [0.0, 0.9, 0.0],
                "distance": -0.001,
            }
        ],
    }

    result = ObstacleAvoidPickTask().evaluate(observation)

    assert result["success"] is False
    assert result["metrics"]["target_grasped"] is True
    assert result["metrics"]["obstacle_contact_force"] == pytest.approx(0.9)


def test_contact_push_succeeds_inside_target_region_under_force_limit() -> None:
    observation = base_observation()
    observation["objects"] = {
        "push_object": {
            "pose": {
                "position": [0.61, 0.02, 0.03],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
            "target_region": {
                "center": [0.6, 0.0, 0.03],
                "radius": 0.05,
            },
        }
    }
    observation["contact"] = {
        "max_force": 0.8,
        "max_penetration": 0.0,
        "contacts": [
            {
                "geom1": "finger_left",
                "geom2": "push_object",
                "position": [0.58, 0.0, 0.03],
                "normal": [1.0, 0.0, 0.0],
                "force": [0.8, 0.0, 0.0],
                "distance": 0.0,
            }
        ],
    }

    result = ContactPushTask().evaluate(observation)

    assert result["success"] is True
    assert result["metrics"]["region_reached"] is True
    assert result["metrics"]["max_contact_force"] == pytest.approx(0.8)


def test_rotate_place_task_checks_position_and_orientation_error() -> None:
    observation = base_observation()
    observation["objects"] = {
        "grasped_object": {
            "pose": {
                "position": [0.4, 0.1, 0.02],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
            "target_pose": {
                "position": [0.405, 0.1, 0.02],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
        }
    }
    observation["robot_state"] = {
        **dict(observation["robot_state"]),
        "grasper_rotation": 1.57,
    }

    result = RotatePlaceTask().evaluate(observation)

    assert result["success"] is True
    assert result["metrics"]["position_error"] == pytest.approx(0.005)
    assert result["metrics"]["orientation_error"] == pytest.approx(0.0)
