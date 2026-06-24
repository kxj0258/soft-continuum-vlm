from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.collect_scripted_demos as collect


class FakeEnv:
    def __init__(self, _config: dict[str, Any] | None = None) -> None:
        self.episode_id = -1
        self.step_id = 0
        self.closed = False

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.episode_id += 1
        self.step_id = 0
        return self._observation([0.0, 0.0, 0.643], grip_command=0.0)

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.step_id += 1
        y = min(0.019, 0.010 * self.step_id)
        observation = self._observation([0.0, y, 0.643], grip_command=action["grip_command"])
        return observation, 0.0, False, {"success": False}

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _observation(tip: list[float], *, grip_command: float) -> dict[str, Any]:
        section_angles = [0.1, 0.0, 0.05, 0.0, 0.02, 0.0]
        return {
            "proprioception": np.asarray([*section_angles, grip_command, 0.0], dtype=np.float32),
            "robot_state": {
                "tip_pose": {"position": tip, "source": "body:feagine_grasper_tip"},
                "section_angles": section_angles,
                "grip_command": grip_command,
                "grasper_rotation": 0.0,
            },
            "contact": {"max_force": 0.0, "max_penetration": 0.0, "contacts": []},
            "language": "pick the red object",
        }


class FakeScriptedExpert:
    position_tolerance = 0.006

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls = 0

    def reset(self, task_name: str | None = None, language: str | None = None) -> None:
        self.calls = 0

    def act(self, observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        phases = ["approach_virtual_object", "close_gripper", "done"]
        distances = [0.02, 0.001, 0.003]
        phase = phases[min(self.calls, len(phases) - 1)]
        distance = distances[min(self.calls, len(distances) - 1)]
        action = {
            "section_angles": [0.1, 0.0, 0.05, 0.0, 0.02, 0.0],
            "grip_command": 1.0 if phase != "approach_virtual_object" else 0.0,
            "grasper_rotation": 0.0,
        }
        info = {
            "phase": phase,
            "target_distance": distance,
            "current_distance_to_target": distance,
            "best_distance_to_target": min(distances[: self.calls + 1]),
            "post_close_drift": 0.002 if phase == "done" else 0.0,
        }
        self.calls += 1
        return action, info


def test_collect_pick_red_object_writes_npz_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pick_red_object_debug.npz"
    monkeypatch.setattr(collect, "FeagineMujocoEnv", FakeEnv)
    monkeypatch.setattr(collect, "ScriptedExpert", FakeScriptedExpert)
    monkeypatch.setattr(
        collect,
        "git_commit_hash",
        lambda: "test-commit",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_scripted_demos.py",
            "--task",
            "pick_red_object",
            "--num-episodes",
            "2",
            "--max-steps",
            "5",
            "--output",
            str(output),
            "--seed",
            "0",
            "--calibration",
            "outputs/calibration/a03_type_2_section_jacobian.json",
            "--target-offset",
            "0.0",
            "0.02",
            "0.0",
            "--close-loop-hold",
            "--close-hold-position-gain-scale",
            "0.35",
            "--max-post-close-drift",
            "0.02",
        ],
    )

    assert collect.main() == 0

    metadata_path = output.with_suffix(".json")
    assert output.exists()
    assert metadata_path.exists()
    with np.load(output, allow_pickle=True) as data:
        fields = set(data.files)
        assert {
            "proprioception",
            "tip_position",
            "section_angles",
            "grip_command",
            "grasper_rotation",
            "action_section_angles",
            "action_grip_command",
            "action_grasper_rotation",
            "language",
            "task_name",
            "phase",
            "target_distance",
            "best_distance_to_target",
            "post_close_drift",
            "reward",
            "done",
            "episode_id",
            "step_id",
            "success",
        }.issubset(fields)
        assert "gripper_rotation" not in fields
        assert data["success"].tolist() == [True] * len(data["success"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["num_episodes"] == 2
    assert metadata["success_rate"] == pytest.approx(1.0)
    assert [episode["success"] for episode in metadata["episodes"]] == [True, True]


def test_collect_rejects_non_pick_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_scripted_demos.py",
            "--task",
            "obstacle_avoid_pick",
            "--output",
            str(tmp_path / "bad.npz"),
        ],
    )

    assert collect.main() != 0

    captured = capsys.readouterr()
    assert (
        "Only pick_red_object is supported by ScriptedExpert demo collection in this narrow step."
        in captured.err
    )
