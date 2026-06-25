from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "diagnose_tabletop_contacts.py"


def _load_module():
    if not SCRIPT_PATH.exists():
        pytest.fail(f"missing script: {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "diagnose_tabletop_contacts",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_mujoco():
    return pytest.importorskip("mujoco", reason="mujoco unavailable")


def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
    except Exception:
        return False
    return True


def _minimal_scene_xml() -> str:
    return """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="mini_contact_scene">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="tabletop_geom" type="box" pos="0 0 -0.025" size="0.2 0.2 0.025" rgba="0.8 0.8 0.8 1"/>
    <body name="feagine_grasper_tip" pos="0.12 0 0.2">
      <geom name="tip_geom" type="sphere" size="0.01" contype="1" conaffinity="1"/>
    </body>
    <body name="red_object" pos="0 0 0.03">
      <freejoint name="red_object_freejoint"/>
      <geom
        name="red_object_geom"
        type="box"
        size="0.025 0.025 0.025"
        mass="0.05"
        friction="0.8 0.1 0.1"
        contype="1"
        conaffinity="1"
        rgba="1 0 0 1"
      />
    </body>
  </worldbody>
</mujoco>
"""


def test_script_module_imports() -> None:
    module = _load_module()
    assert hasattr(module, "diagnose_contacts")
    assert hasattr(module, "write_contact_report")


@pytest.mark.skipif(not _has_mujoco(), reason="mujoco unavailable")
def test_diagnose_contacts_reports_flags_and_serializable_json(tmp_path: Path) -> None:
    _require_mujoco()
    module = _load_module()
    scene_path = tmp_path / "scene.xml"
    output_path = tmp_path / "diagnostics" / "tabletop_contacts.json"
    scene_path.write_text(_minimal_scene_xml(), encoding="utf-8")

    report = module.diagnose_contacts(scene_path=scene_path, steps=200)

    assert "contacts" in report
    assert "pair_summary" in report
    assert "flags" in report
    assert "poses" in report
    assert report["poses"]["red_object"]["position"] is not None
    assert report["poses"]["feagine_grasper_tip"]["position"] is not None
    assert isinstance(report["flags"]["has_any_contact"], bool)
    assert isinstance(report["flags"]["has_red_object_contact"], bool)
    assert isinstance(report["flags"]["has_red_table_contact"], bool)
    assert isinstance(report["flags"]["has_red_grasper_contact"], bool)
    assert isinstance(report["flags"]["has_tip_contact"], bool)
    assert isinstance(report["flags"]["has_obstacle_contact"], bool)

    module.write_contact_report(report, output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert "contacts" in saved
    assert "pair_summary" in saved
    assert "flags" in saved
    assert "poses" in saved

    if saved["num_contacts"] > 0:
        assert any("red_object" in " ".join(item["pair"]) for item in saved["pair_summary"])
