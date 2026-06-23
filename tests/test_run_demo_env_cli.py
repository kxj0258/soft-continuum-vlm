import scripts.run_demo_env as run_demo_env


def test_run_demo_env_headless_overrides_config(monkeypatch) -> None:
    created_configs = []

    class FakeEnv:
        def __init__(self, config):
            created_configs.append(config)
            self.config = type("Config", (), {"render_mode": config["env"]["render_mode"]})()

        def reset(self, **kwargs):
            return {
                "rgb": None,
                "depth": None,
                "proprioception": None,
                "contact": None,
                "language": kwargs.get("language", ""),
            }

        def get_robot_state(self):
            return {"section_angles": (0.0,) * 6}

        def step(self, action):
            return {}, 0.0, True, {"runtime": "fake", "applied_controls": list(action)}

        def close(self):
            pass

    monkeypatch.setattr(run_demo_env, "load_yaml_config", lambda path: {"env": {"render_mode": "human"}})
    monkeypatch.setattr(run_demo_env, "FeagineMujocoEnv", FakeEnv)
    monkeypatch.setattr(run_demo_env.sys, "argv", ["run_demo_env.py", "--headless", "--steps", "1"])

    assert run_demo_env.main() == 0
    assert created_configs[0]["env"]["render_mode"] == "none"
