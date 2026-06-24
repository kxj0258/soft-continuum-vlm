from __future__ import annotations

import json
import subprocess
import sys


def test_run_task_phase_expert_mock_env_writes_rollout_json(tmp_path) -> None:
    output = tmp_path / "mock_obstacle_avoid_pick_expert.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_task_phase_expert.py",
            "--task",
            "obstacle_avoid_pick",
            "--mock-env",
            "--max-steps",
            "20",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["task"] == "obstacle_avoid_pick"
    assert payload["env_type"] == "mock"
    assert payload["steps"]
    assert {"phase", "target_state", "action", "safety", "contact", "reward", "done", "success"} <= set(
        payload["steps"][0]
    )
    assert "metrics" in payload
