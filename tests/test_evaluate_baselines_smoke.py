from __future__ import annotations

import csv
import json
import subprocess
import sys


def test_evaluate_baselines_smoke_writes_json_and_csv(tmp_path) -> None:
    json_output = tmp_path / "baseline.json"
    csv_output = tmp_path / "baseline.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_baselines.py",
            "--tasks",
            "pick_red_object",
            "obstacle_avoid_pick",
            "--baselines",
            "scripted_expert",
            "adapter",
            "vlm_planner_ik",
            "--num-episodes",
            "1",
            "--max-steps",
            "5",
            "--mock-env",
            "--output",
            str(json_output),
            "--csv-output",
            str(csv_output),
        ],
        check=True,
    )

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["env_type"] == "mock"
    assert len(payload["episodes"]) == 6
    assert csv_output.exists()
    assert list(csv.DictReader(csv_output.open(encoding="utf-8")))
