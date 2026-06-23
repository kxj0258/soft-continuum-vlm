import importlib
from types import SimpleNamespace

import pytest


def test_core_research_modules_import_without_feagine() -> None:
    modules = [
        "soft_continuum_vlm.envs.base_env",
        "soft_continuum_vlm.envs.feagine_mujoco_env",
        "soft_continuum_vlm.tasks.base_task",
        "soft_continuum_vlm.tasks.pick_task",
        "soft_continuum_vlm.tasks.obstacle_avoid_pick_task",
        "soft_continuum_vlm.tasks.contact_push_task",
        "soft_continuum_vlm.tasks.rotate_place_task",
        "soft_continuum_vlm.controllers.pcc_ik_controller",
        "soft_continuum_vlm.controllers.scripted_expert",
        "soft_continuum_vlm.perception.object_detector_stub",
        "soft_continuum_vlm.perception.scene_state",
        "soft_continuum_vlm.data.dataset",
        "soft_continuum_vlm.data.replay_buffer",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_feagine_env_explains_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv

    def fail_import(name: str):
        raise ImportError(f"missing {name}")

    env = FeagineMujocoEnv(import_module=fail_import)

    with pytest.raises(ImportError, match="scripts/setup_feagine_env.sh"):
        env.reset()


class FakeData:
    def __init__(self, model: object) -> None:
        self.model = model
        self.qpos = [0.0] * 4
        self.qvel = [0.0] * 4
        self.ctrl = [0.0] * 4
        self.ncon = 0
        self.contact = []


class FakeMujocoModule:
    def __init__(self) -> None:
        self.step_count = 0
        self.forward_count = 0

    def MjData(self, model: object) -> FakeData:
        return FakeData(model)

    def mj_step(self, model: object, data: FakeData) -> None:
        self.step_count += 1

    def mj_forward(self, model: object, data: FakeData) -> None:
        self.forward_count += 1


class FakeRobot:
    def __init__(self, model: object, data: FakeData, *, preset_id: str) -> None:
        self.model = model
        self.data = data
        self.preset_id = preset_id
        self.section_count = 3
        self._section_angles = [0.0] * 6
        self._grip_command = 0.0
        self._grasper_rotation = 0.0
        self.has_grasper_rotation_joint = True

    @property
    def grip_command(self) -> float:
        return self._grip_command

    @property
    def grasper_rotation(self) -> float:
        return self._grasper_rotation

    def get_section_angles(self) -> tuple[float, ...]:
        return tuple(self._section_angles)

    def drive_section_angles(self, values: list[float]) -> None:
        self._section_angles = list(values)

    def set_grip_command(self, command: float) -> None:
        self._grip_command = command

    def drive_grasper_rotation(self, angle: float) -> None:
        self._grasper_rotation = angle

    def tip_pose(self) -> object:
        return SimpleNamespace(position=[1.0, 2.0, 3.0], quaternion=[1.0, 0.0, 0.0, 0.0])


class FakeFeagineMujocoModule:
    def __init__(self) -> None:
        self.load_calls: list[tuple[str, str]] = []
        self.robot: FakeRobot | None = None

    def load_mujoco_model(self, *, preset_id: str, model_type: str) -> object:
        self.load_calls.append((preset_id, model_type))
        return SimpleNamespace(nq=4, nv=4, nu=4)

    def FeagineMjcfRobot(self, model: object, data: FakeData, *, preset_id: str) -> FakeRobot:
        self.robot = FakeRobot(model, data, preset_id=preset_id)
        return self.robot

    def robot_asset_path(self, *, preset_id: str, model_type: str) -> str:
        return f"/fake/{preset_id}/{model_type}.xml"


def test_feagine_env_uses_real_runtime_calls_for_reset_and_step() -> None:
    from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv

    fake_mujoco = FakeMujocoModule()
    fake_feagine = FakeFeagineMujocoModule()

    def import_module(name: str) -> object:
        return {
            "pyfeagine_sim_core": object(),
            "feagine_mujoco": fake_feagine,
            "mujoco": fake_mujoco,
        }[name]

    env = FeagineMujocoEnv(
        {"env": {"robot_preset": "a03_type_2", "asset_model_type": "mjcf", "max_episode_steps": 2}},
        import_module=import_module,
    )
    observation = env.reset(language="pick the red object")

    assert set(observation) == {"rgb", "depth", "proprioception", "contact", "language"}
    assert observation["language"] == "pick the red object"
    assert fake_feagine.load_calls == [("a03_type_2", "mjcf")]

    next_observation, reward, done, info = env.step(
        {
            "section_angles": [0.1, -0.2, 0.11, -0.21, 0.12, -0.22],
            "grip_command": 0.25,
            "grasper_rotation": 0.3,
        }
    )

    assert reward == 0.0
    assert done is False
    assert info["runtime"] == "feagine_mujoco"
    assert info["applied_controls"] == ["section_angles", "grip_command", "grasper_rotation"]
    assert fake_mujoco.step_count == 4
    assert fake_feagine.robot is not None
    assert fake_feagine.robot.get_section_angles() == pytest.approx((0.1, -0.2, 0.11, -0.21, 0.12, -0.22))
    assert fake_feagine.robot.grip_command == pytest.approx(0.25)
    assert fake_feagine.robot.grasper_rotation == pytest.approx(0.3)
    assert next_observation["proprioception"].shape == (8,)
