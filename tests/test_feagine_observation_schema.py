from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from soft_continuum_vlm.data.schema import validate_observation
from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv


class FakeData:
    def __init__(self, model: object) -> None:
        self.model = model
        self.qpos = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        self.qvel = np.asarray([0.0, 0.1, 0.0, -0.1], dtype=np.float64)
        self.xpos = np.asarray([[0.0, 0.0, 0.0], [0.45, 0.0, 0.08]], dtype=np.float64)
        self.xquat = np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self.site_xpos = np.asarray([[0.2, 0.0, 0.2]], dtype=np.float64)
        self.site_xmat = np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        self.ncon = 1
        self.contact = [
            SimpleNamespace(
                geom1=0,
                geom2=1,
                pos=np.asarray([0.4, 0.0, 0.08], dtype=np.float64),
                frame=np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                dist=-0.002,
            )
        ]


class FakeMujocoModule:
    class mjtObj:
        mjOBJ_BODY = 1
        mjOBJ_GEOM = 5
        mjOBJ_SITE = 6

    def MjData(self, model: object) -> FakeData:
        return FakeData(model)

    def mj_step(self, model: object, data: FakeData) -> None:
        pass

    def mj_forward(self, model: object, data: FakeData) -> None:
        pass

    def mj_name2id(self, model: object, obj_type: int, name: str) -> int:
        return model.name_to_id.get((obj_type, name), -1)

    def mj_id2name(self, model: object, obj_type: int, obj_id: int) -> str | None:
        return model.id_to_name.get((obj_type, obj_id), "")

    def mj_contactForce(self, model: object, data: FakeData, contact_index: int, force6d: np.ndarray) -> None:
        force6d[:] = [0.3, 0.4, 0.0, 0.0, 0.0, 0.0]


class FakeRobot:
    section_count = 3

    def __init__(self, model: object, data: FakeData, *, preset_id: str) -> None:
        self._section_angles = [0.1, 0.0, 0.2, 0.0, 0.3, 0.0]
        self.grip_command = 0.25
        self.grasper_rotation = -0.1

    def get_section_angles(self) -> tuple[float, ...]:
        return tuple(self._section_angles)

    def drive_section_angles(self, values: list[float]) -> None:
        self._section_angles = list(values)

    def set_grip_command(self, command: float) -> None:
        self.grip_command = command

    def drive_grasper_rotation(self, angle: float) -> None:
        self.grasper_rotation = angle


class FakeFeagineMujocoModule:
    FeagineMjcfRobot = FakeRobot

    def load_mujoco_model(self, *, preset_id: str, model_type: str) -> object:
        body = FakeMujocoModule.mjtObj.mjOBJ_BODY
        geom = FakeMujocoModule.mjtObj.mjOBJ_GEOM
        site = FakeMujocoModule.mjtObj.mjOBJ_SITE
        return SimpleNamespace(
            nq=4,
            nv=4,
            nu=3,
            nbody=2,
            ngeom=2,
            nsite=1,
            name_to_id={
                (body, "world"): 0,
                (body, "red_object_body"): 1,
                (geom, "finger_tip_geom"): 0,
                (geom, "red_cube_geom"): 1,
                (site, "ee_tip"): 0,
            },
            id_to_name={
                (body, 0): "world",
                (body, 1): "red_object_body",
                (geom, 0): "finger_tip_geom",
                (geom, 1): "red_cube_geom",
                (site, 0): "ee_tip",
            },
            geom_bodyid=np.asarray([0, 1], dtype=np.int32),
        )

    def robot_asset_path(self, *, preset_id: str, model_type: str) -> str:
        return f"/fake/{preset_id}/{model_type}.xml"


def test_feagine_env_fake_runtime_returns_full_structured_observation() -> None:
    fake_mujoco = FakeMujocoModule()
    fake_feagine = FakeFeagineMujocoModule()

    def import_module(name: str) -> object:
        return {
            "pyfeagine_sim_core": object(),
            "feagine_mujoco": fake_feagine,
            "mujoco": fake_mujoco,
        }[name]

    env = FeagineMujocoEnv(
        {
            "env": {
                "render_mode": "none",
                "language": "pick red",
            },
            "scene": {
                "objects": [
                    {
                        "object_id": "red_object",
                        "role": "target",
                        "body_names": ["red_object_body"],
                        "geom_name_patterns": ["red", "cube"],
                    }
                ]
            },
        },
        import_module=import_module,
    )

    observation = env.reset()

    validate_observation(observation)
    assert observation["language"] == "pick red"
    assert observation["robot_state"]["tip_pose"]["position"] == [0.2, 0.0, 0.2]
    assert observation["robot_state"]["tip_pose_source"] == "site:ee_tip"
    assert observation["robot_state"]["section_angles"] == [0.1, 0.0, 0.2, 0.0, 0.3, 0.0]
    assert observation["robot_state"]["qpos"].shape == (4,)
    assert observation["robot_state"]["qvel"].shape == (4,)
    assert observation["objects"]["red_object"]["available"] is True
    assert observation["objects"]["red_object"]["body_name"] == "red_object_body"
    assert observation["contact"]["max_force"] == 0.5
    assert observation["contact"]["max_penetration"] == 0.002
    assert observation["contact"]["robot_contact_count"] == 1
    assert observation["contact"]["target_contact_count"] == 1
