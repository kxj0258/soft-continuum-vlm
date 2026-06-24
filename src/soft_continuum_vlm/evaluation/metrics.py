from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np


def compute_rollout_metrics(step_logs: list[Mapping[str, Any]]) -> dict[str, float]:
    rewards = [float(item.get("reward", 0.0)) for item in step_logs]
    contacts = [item.get("contact", {}) for item in step_logs]
    forces = [float(contact.get("max_force", 0.0)) for contact in contacts if isinstance(contact, Mapping)]
    penetrations = [
        float(contact.get("max_penetration", 0.0)) for contact in contacts if isinstance(contact, Mapping)
    ]
    obstacle_counts = [
        int(contact.get("obstacle_contact_count", 0)) for contact in contacts if isinstance(contact, Mapping)
    ]
    target_counts = [
        int(contact.get("target_contact_count", 0)) for contact in contacts if isinstance(contact, Mapping)
    ]
    self_counts = [
        int(contact.get("robot_self_contact_count", 0)) for contact in contacts if isinstance(contact, Mapping)
    ]
    actions = [item.get("action", item.get("safe_action", {})) for item in step_logs]
    section_norms: list[float] = []
    section_abs_values: list[float] = []
    final_rotation = 0.0
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        section_angles = np.asarray(action.get("section_angles", []), dtype=np.float64).reshape(-1)
        if section_angles.size:
            section_norms.append(float(np.linalg.norm(section_angles)))
            section_abs_values.extend([abs(float(value)) for value in section_angles])
        if "grasper_rotation" in action:
            final_rotation = float(action["grasper_rotation"])
    phases = [str(item.get("phase", "")) for item in step_logs if item.get("phase")]
    safety_clip_count = 0
    safety_block_count = 0
    for item in step_logs:
        safety = item.get("safety", {})
        if not isinstance(safety, Mapping):
            safety = item.get("info", {}).get("safety", {}) if isinstance(item.get("info", {}), Mapping) else {}
        if isinstance(safety, Mapping):
            safety_clip_count += 1 if safety.get("clipped_fields") else 0
            safety_block_count += 1 if safety.get("blocked_fields") else 0
    target_distances = [float(item.get("target_distance", 0.0)) for item in step_logs if "target_distance" in item]
    return {
        "success": 1.0 if any(bool(item.get("success", False)) for item in step_logs) else 0.0,
        "total_reward": float(sum(rewards)),
        "num_steps": float(len(step_logs)),
        "max_contact_force": float(np.max(forces)) if forces else 0.0,
        "mean_contact_force": float(np.mean(forces)) if forces else 0.0,
        "max_penetration": float(np.max(penetrations)) if penetrations else 0.0,
        "mean_penetration": float(np.mean(penetrations)) if penetrations else 0.0,
        "obstacle_contact_count": float(sum(obstacle_counts)),
        "target_contact_count": float(sum(target_counts)),
        "robot_self_contact_count": float(sum(self_counts)),
        "mean_section_angle_norm": float(np.mean(section_norms)) if section_norms else 0.0,
        "max_section_angle_abs": float(np.max(section_abs_values)) if section_abs_values else 0.0,
        "grasper_rotation_final": float(final_rotation),
        "phase_count": float(len(set(phases))),
        "final_target_distance": float(target_distances[-1]) if target_distances else 0.0,
        "safety_clip_count": float(safety_clip_count),
        "safety_block_count": float(safety_block_count),
    }


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
        safety_clip_counts = [float(item.get("safety_clip_count", 0.0)) for item in items]
        safety_block_counts = [float(item.get("safety_block_count", 0.0)) for item in items]
        summary.append(
            {
                "task": task,
                "baseline": baseline,
                "policy": baseline,
                "episodes": len(items),
                "success_rate": float(np.mean(success_values)) if success_values else 0.0,
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "max_contact_force_mean": float(np.mean(max_forces)) if max_forces else 0.0,
                "max_contact_force_max": float(np.max(max_forces)) if max_forces else 0.0,
                "max_penetration_mean": float(np.mean(max_penetrations)) if max_penetrations else 0.0,
                "max_penetration_max": float(np.max(max_penetrations)) if max_penetrations else 0.0,
                "safety_clip_count_mean": float(np.mean(safety_clip_counts)) if safety_clip_counts else 0.0,
                "safety_block_count_mean": float(np.mean(safety_block_counts)) if safety_block_counts else 0.0,
            }
        )
    return summary
