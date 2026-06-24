from __future__ import annotations

import json
import subprocess
import sys

import numpy as np


def test_collect_scripted_demos_records_task_phase_expert_fields(tmp_path) -> None:
    output = tmp_path / "task_phase_demo.npz"

    subprocess.run(
        [
            sys.executable,
            "scripts/collect_scripted_demos.py",
            "--task",
            "obstacle_avoid_pick",
            "--num-episodes",
            "1",
            "--max-steps",
            "8",
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

    data = np.load(output, allow_pickle=True)
    assert "target_state" in data.files
    assert "raw_action" in data.files
    assert "safe_action" in data.files
    assert "safety" in data.files
    assert any("task_phase_expert" in str(item) for item in data["safety"])
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["method_config"]["method"] == "task_phase_expert"
    assert metadata["expert_config"]["controller"] == "PccIkController"
