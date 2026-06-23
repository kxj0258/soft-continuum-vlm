from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")


def test_export_paper_figures_smoke(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    output_dir = tmp_path / "figures"
    metrics.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "task": "pick_red_object",
                        "baseline": "scripted_expert",
                        "success_rate": 1.0,
                        "max_contact_force_mean": 0.2,
                        "max_penetration_mean": 0.0,
                        "episodes": 1,
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
    assert (output_dir / "max_contact_force_by_task.png").exists()
    assert (output_dir / "penetration_by_task.png").exists()
    assert (output_dir / "summary_table.csv").exists()
    assert (output_dir / "summary_table.md").exists()
