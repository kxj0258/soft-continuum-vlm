from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts import run_feagine_metaworld_reach as cli


def _observation(tip=(0.0, 0.0, 0.24)) -> dict[str, Any]:
    return {
        "robot_state": {
            "tip_pose": {"position": list(tip)},
            "section_angles": [0.0] * 6,
            "grip_command": 0.0,
            "grasper_rotation": 0.0,
        },
        "objects": {},
        "contact": {"max_force": 0.0, "max_penetration": 0.0, "contacts": []},
    }


class _MovingBackend:
    def __init__(self) -> None:
        self.tip = np.asarray([0.0, 0.0, 0.24], dtype=np.float64)
        self.last_action: Mapping[str, Any] | None = None

    def reset(self, **_: Any) -> dict[str, Any]:
        self.tip = np.asarray([0.0, 0.0, 0.24], dtype=np.float64)
        return _observation(self.tip.tolist())

    def step(self, action: Mapping[str, Any]):
        self.last_action = dict(action)
        self.tip = np.asarray([0.08, 0.0, 0.24], dtype=np.float64)
        return _observation(self.tip.tolist()), 0.0, False, {"backend": "fake"}

    def render(self):
        return None

    def close(self) -> None:
        return None


class _ProbeBackend(_MovingBackend):
    def probe_tip_position_for_section_angles(self, section_angles):
        values = np.asarray(section_angles, dtype=np.float64)
        return [float(values[0]), float(values[2]), 0.24]


def test_rollout_reach_writes_4d_actions_and_low_level_commands(tmp_path: Path) -> None:
    output = tmp_path / "reach.json"
    backend = _MovingBackend()

    report = cli.run_reach_rollout(
        task_name="feagine_reach_right",
        steps=5,
        seed=0,
        output_path=output,
        backend=backend,
        ik_backend="approximate_pcc",
    )

    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == report
    assert payload["task"] == "feagine_reach_right"
    assert payload["seed"] == 0
    assert payload["steps_requested"] == 5
    assert payload["steps_executed"] == 1
    assert payload["success"] is True
    assert payload["terminated"] is True
    assert payload["truncated"] is False
    assert payload["final_metrics"]["tip_goal_distance"] == 0.0
    assert len(payload["steps"]) == 1
    assert len(payload["steps"][0]["action_4d"]) == 4
    assert set(payload["steps"][0]["runtime_action"]) == {
        "section_angles",
        "grip_command",
        "grasper_rotation",
    }
    assert "ik" in payload["steps"][0]


def test_rollout_can_use_mujoco_fk_ik_backend(tmp_path: Path) -> None:
    output = tmp_path / "reach_mujoco_fk.json"
    backend = _ProbeBackend()

    report = cli.run_reach_rollout(
        task_name="feagine_reach_right",
        steps=5,
        seed=0,
        output_path=output,
        backend=backend,
        ik_backend="mujoco_fk",
    )

    assert report["ik_backend"] == "mujoco_fk"
    assert output.exists()


def test_make_backend_uses_headless_render_mode() -> None:
    captured: dict[str, Any] = {}

    class _Backend:
        def __init__(self, config: Mapping[str, Any]) -> None:
            captured.update(config)

    backend = cli.make_backend(headless=True, backend_type=_Backend)

    assert isinstance(backend, _Backend)
    assert captured["env"]["robot_preset"] == "a03_type_2"
    assert captured["env"]["asset_model_type"] == "mjcf"
    assert captured["env"]["render_mode"] == "none"
