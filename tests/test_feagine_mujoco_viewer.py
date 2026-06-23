import pytest

from types import SimpleNamespace

from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv


class FakeData:
    def __init__(self, model: object) -> None:
        self.model = model
        self.qpos = []
        self.qvel = []
        self.ncon = 0
        self.contact = []


class FakeViewerHandle:
    def __init__(self) -> None:
        self.sync_count = 0
        self.closed = False

    def sync(self) -> None:
        self.sync_count += 1

    def close(self) -> None:
        self.closed = True


class FakeViewerModule:
    def __init__(self) -> None:
        self.handle = FakeViewerHandle()
        self.launch_count = 0

    def launch_passive(self, model: object, data: FakeData) -> FakeViewerHandle:
        self.launch_count += 1
        return self.handle


class FakeMujocoModule:
    def __init__(self) -> None:
        self.viewer = FakeViewerModule()

    def MjData(self, model: object) -> FakeData:
        return FakeData(model)

    def mj_step(self, model: object, data: FakeData) -> None:
        pass

    def mj_forward(self, model: object, data: FakeData) -> None:
        pass


class FakeRobot:
    section_count = 3
    grip_command = 0.0
    grasper_rotation = 0.0

    def __init__(self, model: object, data: FakeData, *, preset_id: str) -> None:
        pass

    def get_section_angles(self) -> tuple[float, ...]:
        return (0.0,) * 6

    def drive_section_angles(self, values: list[float]) -> None:
        pass

    def set_grip_command(self, command: float) -> None:
        self.grip_command = command

    def drive_grasper_rotation(self, angle: float) -> None:
        self.grasper_rotation = angle


class FakeFeagineMujocoModule:
    FeagineMjcfRobot = FakeRobot

    def load_mujoco_model(self, *, preset_id: str, model_type: str) -> object:
        return SimpleNamespace()

    def robot_asset_path(self, *, preset_id: str, model_type: str) -> str:
        return f"/fake/{preset_id}/{model_type}.xml"


def test_default_render_mode_is_human() -> None:
    env = FeagineMujocoEnv()

    assert env.config.render_mode == "human"


def test_human_render_launches_and_closes_passive_viewer() -> None:
    fake_mujoco = FakeMujocoModule()
    fake_feagine = FakeFeagineMujocoModule()

    def import_module(name: str) -> object:
        return {
            "pyfeagine_sim_core": object(),
            "feagine_mujoco": fake_feagine,
            "mujoco": fake_mujoco,
        }[name]

    env = FeagineMujocoEnv({"env": {"render_mode": "human"}}, import_module=import_module)
    env.reset()

    assert fake_mujoco.viewer.launch_count == 1

    env.render()
    assert fake_mujoco.viewer.handle.sync_count == 1

    env.close()
    assert fake_mujoco.viewer.handle.closed is True


def test_none_render_mode_does_not_launch_viewer() -> None:
    fake_mujoco = FakeMujocoModule()
    fake_feagine = FakeFeagineMujocoModule()

    def import_module(name: str) -> object:
        return {
            "pyfeagine_sim_core": object(),
            "feagine_mujoco": fake_feagine,
            "mujoco": fake_mujoco,
        }[name]

    env = FeagineMujocoEnv({"env": {"render_mode": "none"}}, import_module=import_module)
    env.reset()
    env.render()

    assert fake_mujoco.viewer.launch_count == 0


def test_human_render_mode_explains_viewer_start_failure() -> None:
    fake_mujoco = FakeMujocoModule()
    fake_mujoco.viewer.launch_passive = lambda model, data: (_ for _ in ()).throw(RuntimeError("no display"))
    fake_feagine = FakeFeagineMujocoModule()

    def import_module(name: str) -> object:
        return {
            "pyfeagine_sim_core": object(),
            "feagine_mujoco": fake_feagine,
            "mujoco": fake_mujoco,
        }[name]

    env = FeagineMujocoEnv({"env": {"render_mode": "human"}}, import_module=import_module)

    with pytest.raises(RuntimeError, match="--headless"):
        env.reset()
