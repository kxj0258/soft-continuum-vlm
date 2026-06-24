from __future__ import annotations

import pytest

from soft_continuum_vlm.evaluation.metrics import compute_rollout_metrics, summarize_episode_logs, summarize_results


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


def test_compute_rollout_metrics_includes_contact_action_phase_and_safety_counts() -> None:
    step_logs = [
        {
            "reward": -0.1,
            "success": False,
            "phase": "approach",
            "target_distance": 0.2,
            "action": {"section_angles": [0.1, -0.1], "grasper_rotation": 0.0},
            "contact": {
                "max_force": 0.3,
                "max_penetration": 0.0,
                "obstacle_contact_count": 1,
                "target_contact_count": 0,
            },
            "safety": {"clipped_fields": ["section_angles"], "blocked_fields": []},
        },
        {
            "reward": 1.0,
            "success": True,
            "phase": "lift",
            "target_distance": 0.01,
            "action": {"section_angles": [0.2, -0.2], "grasper_rotation": 0.5},
            "contact": {
                "max_force": 0.5,
                "max_penetration": 0.002,
                "obstacle_contact_count": 0,
                "target_contact_count": 1,
            },
            "safety": {"clipped_fields": [], "blocked_fields": ["section_angles"]},
        },
    ]

    metrics = compute_rollout_metrics(step_logs)

    assert metrics["success"] == 1.0
    assert metrics["total_reward"] == pytest.approx(0.9)
    assert metrics["num_steps"] == 2
    assert metrics["max_contact_force"] == pytest.approx(0.5)
    assert metrics["mean_contact_force"] == pytest.approx(0.4)
    assert metrics["max_penetration"] == pytest.approx(0.002)
    assert metrics["obstacle_contact_count"] == 1
    assert metrics["target_contact_count"] == 1
    assert metrics["phase_count"] == 2
    assert metrics["final_target_distance"] == pytest.approx(0.01)
    assert metrics["safety_clip_count"] == 1
    assert metrics["safety_block_count"] == 1
    assert metrics["grasper_rotation_final"] == pytest.approx(0.5)
