from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from soft_continuum_vlm.envs.mujoco_state import (
    body_pose,
    contact_to_dict,
    geom_body_name,
    safe_mj_id2name,
    safe_mj_name2id,
)


class FakeMujoco:
    class mjtObj:
        mjOBJ_BODY = 1
        mjOBJ_GEOM = 5

    def mj_name2id(self, model: object, obj_type: int, name: str) -> int:
        return model.name_to_id.get((obj_type, name), -1)

    def mj_id2name(self, model: object, obj_type: int, obj_id: int) -> str | None:
        return model.id_to_name.get((obj_type, obj_id))

    def mj_contactForce(self, model: object, data: object, contact_index: int, force6d: np.ndarray) -> None:
        force6d[:] = data.contact_forces[contact_index]


def fake_model() -> SimpleNamespace:
    body = FakeMujoco.mjtObj.mjOBJ_BODY
    geom = FakeMujoco.mjtObj.mjOBJ_GEOM
    return SimpleNamespace(
        name_to_id={
            (body, "robot_base"): 0,
            (body, "red_object_body"): 1,
            (geom, "finger_tip_geom"): 0,
            (geom, "red_target_geom"): 1,
        },
        id_to_name={
            (body, 0): "robot_base",
            (body, 1): "red_object_body",
            (geom, 0): "finger_tip_geom",
            (geom, 1): "red_target_geom",
        },
        geom_bodyid=np.asarray([0, 1], dtype=np.int32),
    )


def fake_data() -> SimpleNamespace:
    return SimpleNamespace(
        xpos=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        xquat=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
        contact=[
            SimpleNamespace(
                geom1=0,
                geom2=1,
                pos=np.asarray([0.2, 0.0, 0.1], dtype=np.float64),
                frame=np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
                dist=-0.003,
            )
        ],
        contact_forces=[np.asarray([1.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float64)],
    )


def test_safe_name_helpers_do_not_raise_for_missing_names() -> None:
    mujoco = FakeMujoco()
    model = fake_model()

    assert safe_mj_name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "robot_base") == 0
    assert safe_mj_name2id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "missing") is None
    assert safe_mj_id2name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, 1) == "red_target_geom"
    assert safe_mj_id2name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, 99) == ""


def test_body_pose_and_geom_body_name_use_mujoco_arrays() -> None:
    mujoco = FakeMujoco()
    model = fake_model()
    data = fake_data()

    assert body_pose(model, data, 1) == {
        "position": pytest.approx([0.4, 0.5, 0.6]),
        "orientation": pytest.approx([0.0, 1.0, 0.0, 0.0]),
    }
    assert geom_body_name(mujoco, model, 1) == "red_object_body"


def test_contact_to_dict_preserves_names_force_and_classification() -> None:
    mujoco = FakeMujoco()
    model = fake_model()
    data = fake_data()

    contact = contact_to_dict(
        mujoco,
        model,
        data,
        0,
        name_filters={
            "robot": ["finger", "robot"],
            "target": ["red", "target"],
            "obstacle": ["obstacle"],
        },
    )

    assert contact["geom1_id"] == 0
    assert contact["geom2_id"] == 1
    assert contact["geom1"] == "finger_tip_geom"
    assert contact["geom2"] == "red_target_geom"
    assert contact["body1"] == "robot_base"
    assert contact["body2"] == "red_object_body"
    assert contact["position"] == pytest.approx([0.2, 0.0, 0.1])
    assert contact["normal"] == pytest.approx([0.0, 0.0, 1.0])
    assert contact["force"] == pytest.approx([1.0, 2.0, 2.0])
    assert contact["force_norm"] == pytest.approx(3.0)
    assert contact["distance"] == pytest.approx(-0.003)
    assert contact["is_robot_contact"] is True
    assert contact["is_target_contact"] is True
    assert contact["is_obstacle_contact"] is False
