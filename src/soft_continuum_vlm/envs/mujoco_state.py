from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def safe_mj_name2id(mujoco: Any, model: Any, obj_type: int, name: str) -> int | None:
    """Return a MuJoCo id for a name, or None if the name is unavailable."""

    try:
        obj_id = int(mujoco.mj_name2id(model, obj_type, str(name)))
    except Exception:
        return None
    return None if obj_id < 0 else obj_id


def safe_mj_id2name(mujoco: Any, model: Any, obj_type: int, obj_id: int) -> str:
    """Return a MuJoCo name for an id, or an empty string if unavailable."""

    try:
        name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    except Exception:
        return ""
    return "" if name is None else str(name)


def body_pose(model: Any, data: Any, body_id: int) -> dict[str, list[float]]:
    """Read body world pose from MuJoCo data arrays."""

    _ = model
    position = np.asarray(data.xpos[int(body_id)], dtype=np.float64).reshape(-1)[:3]
    orientation = np.asarray(data.xquat[int(body_id)], dtype=np.float64).reshape(-1)[:4]
    return {
        "position": [float(value) for value in position],
        "orientation": [float(value) for value in orientation],
    }


def geom_body_name(mujoco: Any, model: Any, geom_id: int) -> str:
    geom_index = int(geom_id)
    body_ids = getattr(model, "geom_bodyid", [])
    if geom_index < 0 or geom_index >= len(body_ids):
        return ""
    body_id = int(body_ids[geom_index])
    return safe_mj_id2name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def contact_to_dict(
    mujoco: Any,
    model: Any,
    data: Any,
    contact_index: int,
    *,
    name_filters: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    contact = data.contact[int(contact_index)]
    geom1_id = int(getattr(contact, "geom1", -1))
    geom2_id = int(getattr(contact, "geom2", -1))
    geom1_name = safe_mj_id2name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
    geom2_name = safe_mj_id2name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
    body1_name = geom_body_name(mujoco, model, geom1_id)
    body2_name = geom_body_name(mujoco, model, geom2_id)

    force6d = np.zeros(6, dtype=np.float64)
    if hasattr(mujoco, "mj_contactForce"):
        mujoco.mj_contactForce(model, data, int(contact_index), force6d)
    force = force6d[:3]
    frame = np.asarray(getattr(contact, "frame", np.zeros(9)), dtype=np.float64).reshape(-1)
    normal = frame[:3] if frame.size >= 3 else np.zeros(3, dtype=np.float64)
    position = np.asarray(getattr(contact, "pos", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
    names = [geom1_name, geom2_name, body1_name, body2_name]

    return {
        "geom1": geom1_name,
        "geom2": geom2_name,
        "geom1_id": geom1_id,
        "geom2_id": geom2_id,
        "body1": body1_name,
        "body2": body2_name,
        "position": [float(value) for value in position],
        "normal": [float(value) for value in normal],
        "force": [float(value) for value in force],
        "force_norm": float(np.linalg.norm(force)),
        "distance": float(getattr(contact, "dist", 0.0)),
        "is_robot_contact": _matches_filters(names, name_filters, "robot"),
        "is_obstacle_contact": _matches_filters(names, name_filters, "obstacle"),
        "is_target_contact": _matches_filters(names, name_filters, "target"),
    }


def _matches_filters(names: list[str], name_filters: Mapping[str, list[str]] | None, key: str) -> bool:
    if not name_filters:
        return False
    patterns = [pattern.lower() for pattern in name_filters.get(key, [])]
    lowered_names = [name.lower() for name in names if name]
    return any(pattern in name for pattern in patterns for name in lowered_names)
