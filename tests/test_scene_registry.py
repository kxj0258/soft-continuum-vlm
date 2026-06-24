from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from soft_continuum_vlm.envs.scene_registry import SceneRegistry


class FakeMujoco:
    class mjtObj:
        mjOBJ_BODY = 1
        mjOBJ_GEOM = 5

    def mj_name2id(self, model: object, obj_type: int, name: str) -> int:
        return model.name_to_id.get((obj_type, name), -1)

    def mj_id2name(self, model: object, obj_type: int, obj_id: int) -> str | None:
        return model.id_to_name.get((obj_type, obj_id), "")


def fake_model() -> SimpleNamespace:
    body = FakeMujoco.mjtObj.mjOBJ_BODY
    geom = FakeMujoco.mjtObj.mjOBJ_GEOM
    return SimpleNamespace(
        nbody=2,
        ngeom=2,
        name_to_id={
            (body, "red_object_body"): 1,
            (geom, "red_cube_geom"): 0,
            (geom, "black_obstacle_geom"): 1,
        },
        id_to_name={
            (body, 0): "world",
            (body, 1): "red_object_body",
            (geom, 0): "red_cube_geom",
            (geom, 1): "black_obstacle_geom",
        },
        geom_bodyid=np.asarray([1, 0], dtype=np.int32),
    )


def fake_data() -> SimpleNamespace:
    return SimpleNamespace(
        xpos=np.asarray([[0.0, 0.0, 0.0], [0.3, 0.1, 0.2]], dtype=np.float64),
        xquat=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_scene_registry_resolves_body_names_and_builds_observation() -> None:
    registry = SceneRegistry.from_config(
        {
            "scene": {
                "objects": [
                    {
                        "object_id": "red_object",
                        "role": "target",
                        "color": "red",
                        "shape": "cube",
                        "body_names": ["red_object_body"],
                        "geom_name_patterns": ["red", "cube"],
                    }
                ]
            }
        }
    )

    objects = registry.build_objects_observation(FakeMujoco(), fake_model(), fake_data())

    assert objects["red_object"]["available"] is True
    assert objects["red_object"]["body_name"] == "red_object_body"
    assert objects["red_object"]["geom_names"] == ["red_cube_geom"]
    assert objects["red_object"]["role"] == "target"
    assert objects["red_object"]["pose"]["position"] == pytest.approx([0.3, 0.1, 0.2])


def test_scene_registry_falls_back_to_geom_patterns() -> None:
    registry = SceneRegistry.from_config(
        {
            "scene": {
                "objects": [
                    {
                        "object_id": "obstacle",
                        "role": "obstacle",
                        "body_names": ["missing_body"],
                        "geom_name_patterns": ["black", "obstacle"],
                    }
                ]
            }
        }
    )

    objects = registry.build_objects_observation(FakeMujoco(), fake_model(), fake_data())

    assert objects["obstacle"]["available"] is True
    assert objects["obstacle"]["body_name"] == "world"
    assert objects["obstacle"]["geom_names"] == ["black_obstacle_geom"]


def test_scene_registry_marks_unresolved_objects_available_false() -> None:
    registry = SceneRegistry.from_config(
        {
            "scene": {
                "objects": [
                    {
                        "object_id": "missing_object",
                        "role": "target",
                        "body_names": ["not_in_model"],
                        "geom_name_patterns": ["not_in_model"],
                    }
                ]
            }
        }
    )

    objects = registry.build_objects_observation(FakeMujoco(), fake_model(), fake_data())

    assert objects["missing_object"]["available"] is False
    assert "missing_reason" in objects["missing_object"]
