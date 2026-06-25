from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ENTITY_NAMES = [
    "feagine_grasper_tip",
    "red_object",
    "red_object_geom",
    "blue_object",
    "blue_object_geom",
    "black_obstacle",
    "black_obstacle_geom",
    "target_pad",
    "target_pad_geom",
]

DISTANCE_TARGETS = [
    "red_object",
    "blue_object",
    "black_obstacle",
    "target_pad",
]

GEOM_TYPE_NAMES = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
    8: "sdf",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect tabletop scene entities, distances, and contacts."
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Path to tabletop scene XML.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output diagnostics JSON.",
    )
    return parser.parse_args()


def _to_float_list(value: Any, *, count: int | None = None) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if count is not None:
        array = array[:count]
    return [float(item) for item in array.tolist()]


def _geom_name(model: Any, geom_id: int) -> str:
    name = ""
    try:
        name = str(model.geom(geom_id).name or "")
    except Exception:
        name = ""
    return name if name else f"<unnamed:{int(geom_id)}>"


def _body_name(model: Any, body_id: int) -> str:
    name = ""
    try:
        name = str(model.body(body_id).name or "")
    except Exception:
        name = ""
    return name if name else f"<unnamed:{int(body_id)}>"


def _entity_from_body(model: Any, data: Any, body_id: int, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "body",
        "id": int(body_id),
        "position": _to_float_list(data.xpos[body_id], count=3),
        "orientation": _to_float_list(data.xquat[body_id], count=4),
        "geom_size": None,
        "geom_type": None,
    }


def _entity_from_geom(model: Any, data: Any, geom_id: int, name: str) -> dict[str, Any]:
    geom_type_id = int(model.geom_type[geom_id])
    return {
        "name": name,
        "kind": "geom",
        "id": int(geom_id),
        "position": _to_float_list(data.geom_xpos[geom_id], count=3),
        "orientation": None,
        "geom_size": _to_float_list(model.geom_size[geom_id]),
        "geom_type": GEOM_TYPE_NAMES.get(geom_type_id, str(geom_type_id)),
    }


def _missing_entity(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "missing",
        "id": -1,
        "position": None,
        "orientation": None,
        "geom_size": None,
        "geom_type": None,
    }


def find_named_entity(model: Any, data: Any, name: str) -> dict[str, Any]:
    import mujoco

    try:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    except Exception:
        body_id = -1
    if body_id >= 0:
        return _entity_from_body(model, data, body_id, name)

    try:
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    except Exception:
        geom_id = -1
    if geom_id >= 0:
        return _entity_from_geom(model, data, geom_id, name)

    return _missing_entity(name)


def _distance(position_a: list[float] | None, position_b: list[float] | None) -> float | None:
    if position_a is None or position_b is None:
        return None
    a = np.asarray(position_a, dtype=np.float64)
    b = np.asarray(position_b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def _contact_force6(mujoco: Any, model: Any, data: Any, index: int) -> list[float] | None:
    force = np.zeros(6, dtype=np.float64)
    try:
        mujoco.mj_contactForce(model, data, index, force)
    except Exception:
        return None
    return [float(item) for item in force.tolist()]


def _contact_entry(mujoco: Any, model: Any, data: Any, index: int) -> dict[str, Any]:
    contact = data.contact[index]
    geom1_id = int(contact.geom1)
    geom2_id = int(contact.geom2)
    return {
        "index": int(index),
        "geom1": _geom_name(model, geom1_id),
        "geom2": _geom_name(model, geom2_id),
        "distance": float(contact.dist),
        "position": _to_float_list(contact.pos, count=3),
        "normal": _to_float_list(contact.frame, count=3),
        "force6": _contact_force6(mujoco, model, data, index),
    }


def _contact_labels(model: Any, contact: dict[str, Any]) -> list[str]:
    labels = [str(contact["geom1"]), str(contact["geom2"])]
    for geom_name in (contact["geom1"], contact["geom2"]):
        if not isinstance(geom_name, str) or geom_name.startswith("<unnamed:"):
            continue
        try:
            geom_id = int(model.geom(geom_name).id)
            body_id = int(model.geom_bodyid[geom_id])
        except Exception:
            continue
        labels.append(_body_name(model, body_id))
    return labels


def summarize_contacts(model: Any, contacts: list[dict[str, Any]]) -> dict[str, Any]:
    pair_map: dict[tuple[str, str], dict[str, Any]] = {}
    has_tip_contact = False
    has_red_object_contact = False
    has_obstacle_contact = False

    for contact in contacts:
        pair = tuple(sorted([str(contact["geom1"]), str(contact["geom2"])]))
        force6 = contact.get("force6")
        normal_force = 0.0
        if force6 is not None:
            normal_force = abs(float(force6[0]))

        current = pair_map.get(pair)
        if current is None:
            pair_map[pair] = {
                "pair": [pair[0], pair[1]],
                "count": 1,
                "min_distance": float(contact["distance"]),
                "max_normal_force": float(normal_force),
            }
        else:
            current["count"] = int(current["count"]) + 1
            current["min_distance"] = float(min(float(current["min_distance"]), float(contact["distance"])))
            current["max_normal_force"] = float(max(float(current["max_normal_force"]), float(normal_force)))

        labels = " ".join(label.lower() for label in _contact_labels(model, contact))
        has_tip_contact = has_tip_contact or ("feagine_grasper" in labels or "tip" in labels)
        has_red_object_contact = has_red_object_contact or ("red_object" in labels)
        has_obstacle_contact = has_obstacle_contact or ("black_obstacle" in labels or "obstacle" in labels)

    return {
        "num_contacts": int(len(contacts)),
        "pairs": list(pair_map.values()),
        "has_tip_contact": bool(has_tip_contact),
        "has_red_object_contact": bool(has_red_object_contact),
        "has_obstacle_contact": bool(has_obstacle_contact),
    }


def inspect_scene_state(scene_path: str | Path, *, steps: int) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(resolved_scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(max(0, steps))):
        mujoco.mj_step(model, data)

    entities = {name: find_named_entity(model, data, name) for name in ENTITY_NAMES}
    tip_entity = entities["feagine_grasper_tip"]
    if tip_entity["kind"] == "missing":
        raise ValueError("Missing required entity: feagine_grasper_tip")

    distances_from_tip: dict[str, float | None] = {}
    for name in DISTANCE_TARGETS:
        distances_from_tip[name] = _distance(
            tip_entity["position"],
            entities[name]["position"],
        )

    nearest_name = None
    nearest_distance = math.inf
    for name, distance in distances_from_tip.items():
        if distance is None:
            continue
        if distance < nearest_distance:
            nearest_name = name
            nearest_distance = distance
    nearest_named_object = {
        "name": nearest_name,
        "distance": None if nearest_name is None else float(nearest_distance),
    }

    contacts = [_contact_entry(mujoco, model, data, index) for index in range(int(data.ncon))]
    contact_summary = summarize_contacts(model, contacts)

    return {
        "scene": str(resolved_scene),
        "steps": int(max(0, steps)),
        "entities": entities,
        "distances_from_tip": distances_from_tip,
        "nearest_named_object": nearest_named_object,
        "contacts": contacts,
        "contact_summary": contact_summary,
        "notes": [
            "This script only inspects static scene state and contacts.",
            "It does not make red_object movable or implement grasping.",
        ],
    }


def write_scene_state_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        report = inspect_scene_state(scene_path=scene_path, steps=args.steps)
    except ValueError as exc:
        if "feagine_grasper_tip" in str(exc):
            print("[FAIL] missing required entity: feagine_grasper_tip")
            return 2
        print(f"[FAIL] scene inspection failed: {exc}")
        return 1
    except Exception as exc:
        print(f"[FAIL] scene inspection failed: {exc}")
        return 1

    print(f"[INFO] loaded scene: {scene_path}")
    for name in [
        "feagine_grasper_tip",
        "red_object",
        "blue_object",
        "black_obstacle",
        "target_pad",
    ]:
        entity = report["entities"][name]
        print(f"[ENTITY] {name} kind={entity['kind']} position={entity['position']}")

    for name in DISTANCE_TARGETS:
        print(f"[DIST] tip -> {name} = {report['distances_from_tip'][name]}")

    print(f"[CONTACT] num_contacts={report['contact_summary']['num_contacts']}")
    print(f"[RESULT] nearest_named_object={report['nearest_named_object']}")

    try:
        saved_path = write_scene_state_report(report, output_path)
    except Exception as exc:
        print(f"[FAIL] failed to write diagnostics JSON: {exc}")
        return 3

    print(f"[OK] wrote {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
