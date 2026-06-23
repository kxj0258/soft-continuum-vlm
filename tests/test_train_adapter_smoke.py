from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")


def test_train_adapter_smoke_from_mock_demo(tmp_path) -> None:
    demo = tmp_path / "demo.npz"
    checkpoint = tmp_path / "adapter.pt"
    metrics = tmp_path / "metrics.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/collect_scripted_demos.py",
            "--task",
            "obstacle_avoid_pick",
            "--num-episodes",
            "2",
            "--max-steps",
            "8",
            "--output",
            str(demo),
            "--mock-env",
            "--seed",
            "0",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_adapter.py",
            "--demo",
            str(demo),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--output",
            str(checkpoint),
            "--metrics-output",
            str(metrics),
            "--device",
            "cpu",
        ],
        check=True,
    )

    assert checkpoint.exists()
    saved_metrics = json.loads(metrics.read_text(encoding="utf-8"))
    assert saved_metrics["epochs"] == 2
    assert saved_metrics["final_loss"] >= 0.0
