from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_SCRIPT = REPO_ROOT / "scripts" / "test_red_object_dynamics.py"


def has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not has_mujoco(), reason="MuJoCo is unavailable")
def test_red_object_moves_under_applied_force(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.xml"
    diagnostics_path = tmp_path / "red_object_dynamics.json"
    scene_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="red_object_dynamics_test">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="tabletop_geom" type="box" pos="0 0 -0.025" size="0.2 0.2 0.025" rgba="0.8 0.8 0.8 1"/>
    <body name="red_object" pos="0 0 0.026">
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
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(DYNAMICS_SCRIPT),
            "--scene",
            str(scene_path),
            "--force",
            "1.0",
            "0.0",
            "0.0",
            "--steps",
            "100",
            "--output",
            str(diagnostics_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["red_object"]["has_freejoint"] is True
    assert diagnostics["red_object"]["displacement_norm"] > 1e-4
    assert diagnostics["success"] is True
