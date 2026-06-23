from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def summarize_episode_logs(step_logs: Iterable[Mapping[str, Any]], *, success: bool) -> dict[str, Any]:
    logs = list(step_logs)
    rewards = [float(item.get("reward", 0.0)) for item in logs]
    forces = [float(item.get("contact", {}).get("max_force", 0.0)) for item in logs]
    penetrations = [float(item.get("contact", {}).get("max_penetration", 0.0)) for item in logs]
    return {
        "success": bool(success),
        "total_reward": float(sum(rewards)),
        "num_steps": len(logs),
        "mean_contact_force": float(np.mean(forces)) if forces else 0.0,
        "max_contact_force": float(np.max(forces)) if forces else 0.0,
        "mean_penetration": float(np.mean(penetrations)) if penetrations else 0.0,
        "max_penetration": float(np.max(penetrations)) if penetrations else 0.0,
        "collision_count": int(sum(1 for value in penetrations if value > 0.0)),
        "curvature_or_section_angle_violation_count": 0,
        "final_position_error": 0.0,
        "final_orientation_error": 0.0,
    }


def summarize_results(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(str(result["task"]), str(result["baseline"]))].append(result)
    summary: list[dict[str, Any]] = []
    for (task, baseline), items in sorted(groups.items()):
        success_values = [1.0 if item.get("success") else 0.0 for item in items]
        max_forces = [float(item.get("max_contact_force", 0.0)) for item in items]
        max_penetrations = [float(item.get("max_penetration", 0.0)) for item in items]
        rewards = [float(item.get("total_reward", 0.0)) for item in items]
        summary.append(
            {
                "task": task,
                "baseline": baseline,
                "episodes": len(items),
                "success_rate": float(np.mean(success_values)) if success_values else 0.0,
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "max_contact_force_mean": float(np.mean(max_forces)) if max_forces else 0.0,
                "max_contact_force_max": float(np.max(max_forces)) if max_forces else 0.0,
                "max_penetration_mean": float(np.mean(max_penetrations)) if max_penetrations else 0.0,
                "max_penetration_max": float(np.max(max_penetrations)) if max_penetrations else 0.0,
            }
        )
    return summary
