from __future__ import annotations

import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


def test_generate_tabletop_scene_adds_named_visual_elements_without_changing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_xml = tmp_path / "source_feagine.xml"
    source_text = (
        """<mujoco model="feagine"><asset><mesh name="grasp" file="grasper_rotation.obj"/>"""
        """</asset><worldbody><body name="feagine_grasper_tip"/></worldbody></mujoco>"""
    )
    source_xml.write_text(source_text, encoding="utf-8")
    original_bytes = source_xml.read_bytes()

    fake_feagine_mujoco = types.SimpleNamespace(
        robot_asset_path=lambda preset_id, model_type: str(source_xml)
    )
    monkeypatch.setitem(sys.modules, "feagine_mujoco", fake_feagine_mujoco)

    from scripts.generate_feagine_tabletop_scene import generate_tabletop_scene

    output_xml = tmp_path / "nested" / "feagine_tabletop_scene.xml"
    generated_path = generate_tabletop_scene(output_xml)

    assert generated_path == output_xml
    assert output_xml.exists()
    assert source_xml.read_bytes() == original_bytes

    generated_text = output_xml.read_text(encoding="utf-8")
    for token in (
        "red_object",
        "red_object_geom",
        "blue_object",
        "blue_object_geom",
        "black_obstacle",
        "black_obstacle_geom",
        "target_pad",
        "target_pad_geom",
        "camera",
        "light",
    ):
        assert token in generated_text

    root = ET.parse(output_xml).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    assert compiler.attrib["meshdir"] == source_xml.parent.as_posix()
    assert compiler.attrib["texturedir"] == source_xml.parent.as_posix()

    worldbody = root.find("worldbody")
    assert worldbody is not None

    bodies = {
        body.attrib.get("name"): body
        for body in worldbody.findall("body")
        if body.attrib.get("name")
    }

    red_body = bodies["red_object"]
    red_freejoint = red_body.find("freejoint")
    assert red_freejoint is not None
    assert red_freejoint.attrib.get("name") == "red_object_freejoint"

    red_geom = red_body.find("geom")
    assert red_geom is not None
    assert red_geom.attrib.get("name") == "red_object_geom"
    assert red_geom.attrib.get("mass") or red_body.find("inertial") is not None
    assert red_geom.attrib.get("friction") == "0.8 0.1 0.1"
    assert red_geom.attrib.get("contype", "1") != "0"
    assert red_geom.attrib.get("conaffinity", "1") != "0"

    for fixed_name in ("blue_object", "black_obstacle"):
        assert bodies[fixed_name].find("freejoint") is None


def test_generate_tabletop_scene_reachable_variant_adds_pedestal_and_keeps_red_object_freejoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_xml = tmp_path / "source_feagine.xml"
    source_xml.write_text(
        """<mujoco model="feagine"><asset></asset><worldbody><body name="feagine_grasper_tip"/></worldbody></mujoco>""",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        sys.modules,
        "feagine_mujoco",
        types.SimpleNamespace(robot_asset_path=lambda preset_id, model_type: str(source_xml)),
    )

    from scripts.generate_feagine_tabletop_scene import generate_tabletop_scene, main

    default_xml = tmp_path / "default.xml"
    generate_tabletop_scene(default_xml, variant="default")

    output_xml = tmp_path / "reachable.xml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_feagine_tabletop_scene.py",
            "--variant",
            "reachable",
            "--output",
            str(output_xml),
        ],
    )

    assert main() == 0

    default_root = ET.parse(default_xml).getroot()
    reachable_root = ET.parse(output_xml).getroot()
    reachable_red = reachable_root.find("./worldbody/body[@name='red_object']")
    reachable_pedestal = reachable_root.find("./worldbody/body[@name='red_pedestal']")
    reachable_red_geom = reachable_root.find("./worldbody/body[@name='red_object']/geom[@name='red_object_geom']")
    reachable_pedestal_geom = reachable_root.find("./worldbody/body[@name='red_pedestal']/geom[@name='red_pedestal_geom']")
    default_red = default_root.find("./worldbody/body[@name='red_object']")
    blue_body = reachable_root.find("./worldbody/body[@name='blue_object']")
    obstacle_body = reachable_root.find("./worldbody/body[@name='black_obstacle']")

    assert default_red is not None
    assert reachable_red is not None
    assert reachable_red.find("freejoint") is not None
    assert reachable_red_geom is not None
    assert reachable_pedestal is not None
    assert reachable_pedestal_geom is not None
    assert reachable_pedestal.find("freejoint") is None
    assert blue_body is not None
    assert blue_body.find("freejoint") is None
    assert obstacle_body is not None
    assert obstacle_body.find("freejoint") is None

    default_red_z = float(default_red.attrib["pos"].split()[2])
    reachable_red_z = float(reachable_red.attrib["pos"].split()[2])
    assert reachable_red_z > default_red_z
