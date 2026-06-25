from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "inspect_tabletop_scene_state.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "inspect_tabletop_scene_state",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_mujoco():
    return pytest.importorskip("mujoco", reason="mujoco unavailable")


def _minimal_scene_xml(*, include_tip: bool = True) -> str:
    tip_body = (
        '<body name="feagine_grasper_tip" pos="0 0 0">'
        '<geom name="tip_geom" type="sphere" size="0.01"/>'
        "</body>"
        if include_tip
        else ""
    )
    return f"""<mujoco model="mini_scene">
  <worldbody>
    {tip_body}
    <body name="red_object" pos="0.1 0 0">
      <geom name="red_object_geom" type="box" size="0.01 0.01 0.01"/>
    </body>
    <body name="blue_object" pos="0 0.2 0">
      <geom name="blue_object_geom" type="sphere" size="0.02"/>
    </body>
    <body name="black_obstacle" pos="0 0 0.3">
      <geom name="black_obstacle_geom" type="box" size="0.02 0.02 0.02"/>
    </body>
    <geom name="target_pad" type="cylinder" pos="0.05 0.05 0" size="0.03 0.01"/>
    <geom name="target_pad_geom" type="cylinder" pos="0.05 0.05 0" size="0.03 0.01"/>
  </worldbody>
</mujoco>
"""


def test_script_module_imports() -> None:
    module = _load_module()
    assert hasattr(module, "inspect_scene_state")
    assert hasattr(module, "write_scene_state_report")


def test_inspect_scene_reads_body_and_geom_entities(tmp_path: Path) -> None:
    _require_mujoco()
    module = _load_module()
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text(_minimal_scene_xml(), encoding="utf-8")

    report = module.inspect_scene_state(scene_path=scene_path, steps=2)

    assert report["entities"]["red_object"]["kind"] == "body"
    assert report["entities"]["red_object_geom"]["kind"] == "geom"
    assert report["entities"]["target_pad"]["kind"] == "geom"


def test_inspect_scene_computes_tip_distance(tmp_path: Path) -> None:
    _require_mujoco()
    module = _load_module()
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text(_minimal_scene_xml(), encoding="utf-8")

    report = module.inspect_scene_state(scene_path=scene_path, steps=0)

    assert report["distances_from_tip"]["red_object"] == pytest.approx(0.1)


def test_inspect_scene_writes_json(tmp_path: Path) -> None:
    _require_mujoco()
    module = _load_module()
    scene_path = tmp_path / "scene.xml"
    output_path = tmp_path / "diagnostics" / "scene_state.json"
    scene_path.write_text(_minimal_scene_xml(), encoding="utf-8")

    report = module.inspect_scene_state(scene_path=scene_path, steps=1)
    module.write_scene_state_report(report, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert "entities" in saved
    assert "distances_from_tip" in saved
    assert "contact_summary" in saved


def test_inspect_scene_raises_when_tip_missing(tmp_path: Path) -> None:
    _require_mujoco()
    module = _load_module()
    scene_path = tmp_path / "scene.xml"
    scene_path.write_text(_minimal_scene_xml(include_tip=False), encoding="utf-8")

    with pytest.raises(ValueError, match="feagine_grasper_tip"):
        module.inspect_scene_state(scene_path=scene_path, steps=0)
