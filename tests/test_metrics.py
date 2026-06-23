from __future__ import annotations

import pytest

from soft_continuum_vlm.evaluation.metrics import summarize_episode_logs, summarize_results


def test_summarize_episode_logs_computes_contact_metrics() -> None:
    logs = [
        {"reward": 1.0, "contact": {"max_force": 0.2, "max_penetration": 0.0}},
        {"reward": -0.5, "contact": {"max_force": 0.4, "max_penetration": 0.01}},
    ]

    metrics = summarize_episode_logs(logs, success=True)

    assert metrics["success"] is True
    assert metrics["total_reward"] == pytest.approx(0.5)
    assert metrics["mean_contact_force"] == pytest.approx(0.3)
    assert metrics["max_contact_force"] == pytest.approx(0.4)
    assert metrics["mean_penetration"] == pytest.approx(0.005)
    assert metrics["collision_count"] == 1


def test_summarize_results_groups_by_task_and_baseline() -> None:
    results = [
        {"task": "pick", "baseline": "scripted", "success": True, "max_contact_force": 0.2, "max_penetration": 0.0},
        {"task": "pick", "baseline": "scripted", "success": False, "max_contact_force": 0.4, "max_penetration": 0.01},
    ]

    summary = summarize_results(results)

    assert summary[0]["task"] == "pick"
    assert summary[0]["baseline"] == "scripted"
    assert summary[0]["success_rate"] == pytest.approx(0.5)
    assert summary[0]["episodes"] == 2
