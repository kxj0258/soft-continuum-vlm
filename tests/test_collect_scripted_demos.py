from __future__ import annotations

import json
import subprocess
import sys

import numpy as np


def test_collect_scripted_demos_mock_env_creates_npz_and_metadata(tmp_path) -> None:
    output = tmp_path / "debug_obstacle_avoid_pick.npz"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/collect_scripted_demos.py",
            "--task",
            "obstacle_avoid_pick",
            "--num-episodes",
            "2",
            "--max-steps",
            "6",
            "--output",
            str(output),
            "--mock-env",
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert "saved" in result.stdout.lower()
    data = np.load(output, allow_pickle=True)
    assert data["action_vector"].shape[0] > 0
    assert data["proprioception"].shape[0] == data["action_vector"].shape[0]
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["env_type"] == "mock"
    assert metadata["action_schema"] == ["section_angles", "grip_command", "grasper_rotation"]
