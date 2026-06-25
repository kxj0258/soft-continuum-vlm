from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def test_validate_reachable_scene_reports_pedestal_contact_and_writes_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mujoco = pytest.importorskip("mujoco")

    source_xml = tmp_path / "source_feagine.xml"
    source_xml.write_text(
        """
        <mujoco model="feagine">
          <asset></asset>
          <worldbody>
            <body name="feagine_grasper_tip" pos="0.3 0.0 0.5">
              <geom name="tip_geom" type="sphere" size="0.01" />
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    monkeypatch.setitem(
        sys.modules,
        "feagine_mujoco",
        types.SimpleNamespace(robot_asset_path=lambda preset_id, model_type: str(source_xml)),
    )

    from scripts.generate_feagine_tabletop_scene import generate_tabletop_scene
    from scripts.validate_reachable_scene import validate_reachable_scene, write_report

    scene_path = generate_tabletop_scene(tmp_path / "reachable.xml", variant="reachable")
    report = validate_reachable_scene(scene_path, steps=10)

    assert report["red_object"]["has_freejoint"] is True
    assert report["red_pedestal"]["exists"] is True
    assert report["flags"]["has_red_pedestal_contact"] in {True, False}
    assert report["red_object"]["position"][2] >= 0.28

    output = write_report(report, tmp_path / "report.json")
    assert output.exists()
    assert output.read_text(encoding="utf-8")
