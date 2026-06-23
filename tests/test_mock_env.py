from __future__ import annotations

from soft_continuum_vlm.envs.mock_env import MockContinuumEnv


def test_mock_env_observation_schema_and_deterministic_step() -> None:
    env = MockContinuumEnv(task="obstacle_avoid_pick")
    obs = env.reset(language="Reach around the obstacle.", seed=7)

    assert obs["rgb"].shape == (64, 64, 3)
    assert obs["depth"].shape == (64, 64)
    assert set(obs["robot_state"]) >= {"tip_pose", "section_angles", "grip_command", "grasper_rotation"}
    assert "target_object" in obs["objects"]
    assert set(obs["contact"]) >= {"max_force", "max_penetration", "contacts"}

    next_obs, reward, done, info = env.step(
        {
            "section_angles": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        }
    )

    assert next_obs["robot_state"]["tip_pose"]["position"] != obs["robot_state"]["tip_pose"]["position"]
    assert isinstance(reward, float)
    assert done is False
    assert info["task_name"] == "obstacle_avoid_pick"


def test_mock_env_grasps_target_when_close_and_grip_closes() -> None:
    env = MockContinuumEnv(task="pick_red_object")
    env.reset(seed=0)

    for _ in range(10):
        obs, _, _, _ = env.step(
            {
                "section_angles": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "grip_command": 1.0,
                "grasper_rotation": 0.0,
            }
        )

    assert obs["objects"]["red_object"]["grasped"] is True
