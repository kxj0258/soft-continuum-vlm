from __future__ import annotations

from soft_continuum_vlm.planners.deterministic_vlm_planner import DeterministicVLMPlanner


def test_deterministic_vlm_planner_outputs_top_level_task_flags() -> None:
    planner = DeterministicVLMPlanner()

    output = planner.plan(
        "绕过黑色障碍物，轻轻旋转并放置红色物体",
        observation={"objects": {"red_object": {}, "obstacle": {}}},
        task_name="rotate_and_place",
    )

    assert output["target_object"] == "red_object"
    assert output["requires_rotation"] is True
    assert output["requires_push"] is False
    assert output["grasp_mode"] == "gentle"
    assert output["avoid_objects"]
