from __future__ import annotations

from soft_continuum_vlm.planners.deterministic_vlm_planner import DeterministicVLMPlanner


def test_planner_parses_chinese_color_avoid_and_gentle_grasp() -> None:
    planner = DeterministicVLMPlanner()

    output = planner.plan(
        "绕过黑色障碍物，轻轻抓住蓝色圆柱",
        observation={"objects": {"blue_object": {}, "black_obstacle": {}}},
        task_name="obstacle_avoid_pick",
    )

    assert output["target_object"] == "blue_object"
    assert output["avoid_objects"] == ["black_obstacle"]
    assert output["grasp_mode"] == "gentle"
    assert output["contact_force_limit"] < 1.0
    assert output["language_constraints"]["avoid_collision"] is True
    assert output["language_constraints"]["gentle_contact"] is True
    assert [item["phase"] for item in output["subgoals"]][:3] == ["approach", "avoid", "grasp"]


def test_planner_parses_english_rotation_and_fallback_target() -> None:
    planner = DeterministicVLMPlanner()

    output = planner.plan(
        "rotate and place the yellow object gently",
        observation={"objects": {}},
        task_name="rotate_and_place",
    )

    assert output["target_object"] == "yellow_object"
    assert output["language_constraints"]["requires_rotation"] is True
    assert output["grasp_mode"] == "gentle"


def test_planner_falls_back_for_uncertain_language() -> None:
    planner = DeterministicVLMPlanner()

    output = planner.plan("", observation={"objects": {"red_object": {}}}, task_name="pick_red_object")

    assert output["target_object"] == "red_object"
    assert output["approach_side"] == "any"
    assert output["subgoals"]
