from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")


def test_evaluate_adapter_smoke_from_trained_checkpoint(tmp_path) -> None:
    demo = tmp_path / "demo.npz"
    checkpoint = tmp_path / "adapter.pt"
    train_metrics = tmp_path / "train_metrics.json"
    eval_metrics = tmp_path / "eval_metrics.json"

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
            "1",
            "--batch-size",
            "4",
            "--output",
            str(checkpoint),
            "--metrics-output",
            str(train_metrics),
            "--device",
            "cpu",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_adapter.py",
            "--checkpoint",
            str(checkpoint),
            "--task",
            "obstacle_avoid_pick",
            "--num-episodes",
            "2",
            "--max-steps",
            "6",
            "--mock-env",
            "--output",
            str(eval_metrics),
        ],
        check=True,
    )

    metrics = json.loads(eval_metrics.read_text(encoding="utf-8"))
    assert metrics["num_episodes"] == 2
    assert metrics["num_steps"] > 0
    assert 0.0 <= metrics["success_rate"] <= 1.0
