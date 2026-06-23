from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ACTION_KEYS = ("section_angles", "grip_command", "grasper_rotation")
OPTIONAL_ACTION_KEYS = ("joint_targets", "segment_joint_targets")
OBSERVATION_KEYS = (
    "rgb",
    "depth",
    "language",
    "proprioception",
    "robot_state",
    "objects",
    "contact",
)
CONTACT_KEYS = ("max_force", "max_penetration", "contacts")
ROBOT_STATE_KEYS = ("tip_pose", "section_angles", "grip_command", "grasper_rotation")
DATASET_REQUIRED_KEYS = (
    "proprioception",
    "contact",
    "language",
    "language_feature",
    "morphology",
    "action",
    "action_vector",
    "reward",
    "done",
    "success",
    "task_name",
    "phase",
    "episode_id",
    "step_id",
)
OBSERVATION_SCHEMA_VERSION = "structured-v1"


def validate_action(action: Mapping[str, Any]) -> None:
    if "gripper_rotation" in action:
        raise ValueError("Use 'grasper_rotation' instead of unsupported 'gripper_rotation'.")
    allowed = set(ACTION_KEYS + OPTIONAL_ACTION_KEYS)
    unknown = sorted(set(action) - allowed)
    if unknown:
        raise ValueError(f"Unsupported action fields: {unknown}. Allowed fields: {sorted(allowed)}")
    for key in ACTION_KEYS:
        if key not in action:
            raise ValueError(f"Action is missing required field '{key}'.")


def validate_observation(obs: Mapping[str, Any]) -> None:
    missing = [key for key in OBSERVATION_KEYS if key not in obs]
    if missing:
        raise ValueError(f"Observation is missing required field(s): {missing}")
    contact = obs.get("contact", {})
    if not isinstance(contact, Mapping):
        raise ValueError("Observation field 'contact' must be a mapping.")
    missing_contact = [key for key in CONTACT_KEYS if key not in contact]
    if missing_contact:
        raise ValueError(f"Contact is missing required field(s): {missing_contact}")
    robot_state = obs.get("robot_state", {})
    if not isinstance(robot_state, Mapping):
        raise ValueError("Observation field 'robot_state' must be a mapping.")
    missing_robot = [key for key in ROBOT_STATE_KEYS if key not in robot_state]
    if missing_robot:
        raise ValueError(f"Robot state is missing required field(s): {missing_robot}")


def flatten_action(action: Mapping[str, Any], section_count: int | None = None) -> np.ndarray:
    validate_action(action)
    section_angles = np.asarray(action["section_angles"], dtype=np.float32).reshape(-1)
    expected = 2 * section_count if section_count is not None else section_angles.size
    if section_angles.size != expected:
        raise ValueError(f"Expected {expected} section angle values, got {section_angles.size}.")
    tail = np.asarray([float(action["grip_command"]), float(action["grasper_rotation"])], dtype=np.float32)
    return np.concatenate([section_angles.astype(np.float32), tail])


def unflatten_action(vector: np.ndarray, section_count: int) -> dict[str, Any]:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    expected = 2 * section_count + 2
    if values.size != expected:
        raise ValueError(f"Expected {expected} action values, got {values.size}.")
    return {
        "section_angles": [float(value) for value in values[: 2 * section_count]],
        "grip_command": float(values[-2]),
        "grasper_rotation": float(values[-1]),
    }


def flatten_contact(contact: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(contact.get("max_force", 0.0)),
            float(contact.get("max_penetration", 0.0)),
            float(len(contact.get("contacts", []) or [])),
        ],
        dtype=np.float32,
    )


def flatten_proprioception(obs: Mapping[str, Any]) -> np.ndarray:
    raw = obs.get("proprioception")
    if raw is not None:
        return np.asarray(raw, dtype=np.float32).reshape(-1)
    robot_state = obs.get("robot_state", {})
    if not isinstance(robot_state, Mapping):
        return np.zeros(0, dtype=np.float32)
    return np.asarray(
        [
            *list(robot_state.get("section_angles", [])),
            float(robot_state.get("grip_command", 0.0)),
            float(robot_state.get("grasper_rotation", 0.0)),
        ],
        dtype=np.float32,
    )
