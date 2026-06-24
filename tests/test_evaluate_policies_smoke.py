from __future__ import annotations

import csv
import json
import subprocess
import sys


def test_evaluate_policies_mock_smoke_writes_json_and_csv(tmp_path) -> None:
    output = tmp_path / "policy_eval.json"
    csv_output = tmp_path / "policy_eval.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_policies.py",
            "--tasks",
            "pick_red_object",
            "obstacle_avoid_pick",
            "--policies",
            "task_phase_expert",
            "vlm_planner_ik",
            "--mock-env",
            "--num-episodes",
            "1",
            "--max-steps",
            "12",
            "--seed",
            "0",
            "--output",
            str(output),
            "--csv-output",
            str(csv_output),
        ],
        check=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["env_type"] == "mock"
    assert len(payload["episodes"]) == 4
    assert payload["metadata"]["action_schema"] == ["section_angles", "grip_command", "grasper_rotation"]
    rows = list(csv.DictReader(csv_output.open(encoding="utf-8")))
    assert rows
    assert {"task", "policy", "seed", "success", "total_reward", "num_steps", "language"} <= set(rows[0])
