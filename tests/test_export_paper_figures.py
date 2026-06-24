from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")


def test_export_paper_figures_writes_policy_plots_and_tables(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    output_dir = tmp_path / "figures"
    metrics.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "task": "pick_red_object",
                        "baseline": "task_phase_expert",
                        "policy": "task_phase_expert",
                        "episodes": 1,
                        "success_rate": 1.0,
                        "mean_reward": 0.5,
                        "max_contact_force_mean": 0.2,
                        "max_penetration_mean": 0.0,
                        "safety_clip_count_mean": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/export_paper_figures.py",
            "--metrics",
            str(metrics),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    assert (output_dir / "success_rate_by_task.png").exists()
    assert (output_dir / "contact_force_by_policy.png").exists()
    assert (output_dir / "penetration_by_policy.png").exists()
    assert (output_dir / "safety_clip_count_by_policy.png").exists()
    assert (output_dir / "summary_table.csv").exists()
    assert (output_dir / "summary_table.md").exists()
