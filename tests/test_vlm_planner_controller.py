from __future__ import annotations

from soft_continuum_vlm.controllers.safety_projector import SafetyLimits, SafetyProjector
from soft_continuum_vlm.controllers.vlm_planner_controller import VlmPlannerController
from soft_continuum_vlm.envs.mock_env import MockContinuumEnv


def test_vlm_planner_controller_logs_planner_and_safety_info() -> None:
    env = MockContinuumEnv(task="obstacle_avoid_pick")
    observation = env.reset(language="绕过黑色障碍物，轻轻抓住蓝色圆柱", seed=0)
    controller = VlmPlannerController(
        safety_projector=SafetyProjector(
            SafetyLimits(
                max_abs_section_angle=0.2,
                max_gripper_rotation=0.5,
                max_contact_force=0.8,
                max_penetration=0.01,
            )
        )
    )

    action, info = controller.act(
        language="绕过黑色障碍物，轻轻抓住蓝色圆柱",
        observation=observation,
        task_name="obstacle_avoid_pick",
    )

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert info["planner_output"]["target_object"] == "blue_object"
    assert info["selected_subgoal"]["phase"] == "approach"
    assert info["phase"] == "approach"
    assert set(info) >= {"safety_info", "raw_action", "safe_action"}
