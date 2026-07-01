from __future__ import annotations

import numpy as np
import pytest

from soft_continuum_vlm.controllers.pick_place_expert import FeaginePickPlaceExpert
from soft_continuum_vlm.tasks.feagine_pick_place_tasks import (
    FeaginePickLeftPlaceRightTask,
    FeaginePickRightPlaceLeftTask,
    FeaginePickShelfPlaceShelfTask,
)


def _observation(
    *,
    tip=(0.0, 0.0, 0.2),
    object_position=(-0.08, 0.0, 0.2),
    grip_command=0.0,
    grasper_rotation=0.0,
    grasped=False,
):
    return {
        "robot_state": {
            "tip_pose": {"position": list(tip)},
            "section_angles": [0.0] * 6,
            "grip_command": grip_command,
            "grasper_rotation": grasper_rotation,
        },
        "objects": {
            "pick_object": {
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


def test_pick_place_requires_full_grasp_lift_place_and_retract_sequence() -> None:
    goal = [0.08, 0.0, 0.2]
    task = FeaginePickLeftPlaceRightTask(
        goal_position=goal,
        lift_height=0.06,
        approach_threshold=0.02,
        place_threshold=0.025,
        retract_distance=0.05,
    )
    initial = _observation()
    task.reset_task(observation=initial)

    task.evaluate(_observation(tip=[-0.08, 0.0, 0.2]))
    assert task.phase == "align_grasper"
    task.evaluate(_observation(tip=[-0.08, 0.0, 0.2], grasper_rotation=0.0))
    assert task.phase == "close_gripper"
    task.evaluate(_observation(tip=[-0.08, 0.0, 0.2], grip_command=1.0, grasped=True))
    assert task.phase == "lift"
    task.evaluate(
        _observation(
            tip=[-0.08, 0.0, 0.26],
            object_position=[-0.08, 0.0, 0.26],
            grip_command=1.0,
            grasped=True,
        )
    )
    assert task.phase == "transport"
    task.evaluate(
        _observation(
            tip=[0.08, 0.0, 0.26],
            object_position=[0.08, 0.0, 0.26],
            grip_command=1.0,
            grasped=True,
        )
    )
    assert task.phase == "align_place"
    task.evaluate(
        _observation(
            tip=goal,
            object_position=goal,
            grip_command=1.0,
            grasper_rotation=0.0,
            grasped=True,
        )
    )
    assert task.phase == "release"
    task.evaluate(
        _observation(
            tip=goal,
            object_position=goal,
            grip_command=0.0,
            grasped=False,
        )
    )
    assert task.phase == "retract"
    result = task.evaluate(
        _observation(
            tip=[0.08, 0.0, 0.27],
            object_position=goal,
            grip_command=0.0,
            grasped=False,
        )
    )

    assert task.phase == "complete"
    assert result.success is True
    assert result.metrics["grasp_success"] == 1.0
    assert result.metrics["lift_success"] == 1.0
    assert result.metrics["place_success"] == 1.0


@pytest.mark.parametrize(
    "task_type,name",
    [
        (FeaginePickLeftPlaceRightTask, "feagine_pick_left_place_right"),
        (FeaginePickRightPlaceLeftTask, "feagine_pick_right_place_left"),
        (FeaginePickShelfPlaceShelfTask, "feagine_pick_shelf_place_shelf"),
    ],
)
def test_pick_place_task_variants_have_stable_names(task_type, name) -> None:
    assert task_type().name == name


def test_pick_place_context_exposes_axes_without_top_level_rotation_action() -> None:
    task = FeaginePickLeftPlaceRightTask()
    observation = _observation()
    task.reset_task(observation=observation)
    task.phase = "align_grasper"

    context = task.get_task_context(observation)

    assert context["phase"] == "align_grasper"
    assert context["object_principal_axis"] == [1.0, 0.0, 0.0]


def test_closed_gripper_without_grasped_object_does_not_advance() -> None:
    task = FeaginePickLeftPlaceRightTask()
    observation = _observation(grip_command=1.0, grasped=False)
    task.reset_task(observation=observation)
    task.phase = "close_gripper"

    result = task.evaluate(observation)

    assert task.phase == "close_gripper"
    assert result.metrics["grasp_success"] == 0.0
    assert result.success is False


def test_pick_place_expert_emits_4d_gripper_commands_by_phase() -> None:
    expert = FeaginePickPlaceExpert(delta_xyz_scale=0.01)
    observation = _observation(tip=[-0.08, 0.0, 0.2])
    observation["task"] = {
        "phase": "close_gripper",
        "object_name": "pick_object",
        "goal_position": [0.08, 0.0, 0.2],
        "lift_height": 0.06,
    }

    close_action = expert.act(observation)
    observation["task"]["phase"] = "release"
    release_action = expert.act(observation)

    assert close_action.shape == (4,)
    np.testing.assert_allclose(close_action[:3], [0.0, 0.0, 0.0])
    assert close_action[3] == pytest.approx(1.0)
    assert release_action[3] == pytest.approx(-1.0)
