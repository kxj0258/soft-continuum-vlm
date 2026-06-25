from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose tabletop contacts after passive settling.")
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--steps", type=int, default=200, help="Number of passive settling steps.")
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    return parser.parse_args()


def _to_float_list(values: Any, *, count: int | None = None) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
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


def _find_body_id(mujoco: Any, model: Any, name: str) -> int:
    try:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    except Exception:
        return -1


def _find_geom_id(mujoco: Any, model: Any, name: str) -> int:
    try:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    except Exception:
        return -1


def _entity_pose(model: Any, data: Any, *, body_name: str | None = None, geom_name: str | None = None) -> dict[str, Any]:
    if body_name is not None:
        try:
            body_id = int(model.body(body_name).id)
            return {
                "position": _to_float_list(data.xpos[body_id], count=3),
                "orientation": _to_float_list(data.xquat[body_id], count=4),
            }
        except Exception:
            pass
    if geom_name is not None:
        try:
            geom_id = int(model.geom(geom_name).id)
            return {
                "position": _to_float_list(data.geom_xpos[geom_id], count=3),
                "orientation": None,
            }
        except Exception:
            pass
    return {"position": None, "orientation": None}


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
    body1_id = int(model.geom_bodyid[geom1_id])
    body2_id = int(model.geom_bodyid[geom2_id])
    force6 = _contact_force6(mujoco, model, data, index)
    normal_force = None if force6 is None else float(force6[0])
    return {
        "index": int(index),
        "geom1_id": geom1_id,
        "geom2_id": geom2_id,
        "geom1": _geom_name(model, geom1_id),
        "geom2": _geom_name(model, geom2_id),
        "body1": _body_name(model, body1_id),
        "body2": _body_name(model, body2_id),
        "distance": float(contact.dist),
        "position": _to_float_list(contact.pos, count=3),
        "normal": _to_float_list(contact.frame, count=3),
        "force6": force6,
        "normal_force": normal_force,
    }


def _pair_summary(contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for contact in contacts:
        geom_pair = tuple(sorted([str(contact["geom1"]), str(contact["geom2"])]))
        body_pair = tuple(sorted([str(contact["body1"]), str(contact["body2"])]))
        current = summary.get(geom_pair)
        normal_force = contact.get("normal_force")
        abs_normal_force = None if normal_force is None else abs(float(normal_force))
        if current is None:
            summary[geom_pair] = {
                "pair": [geom_pair[0], geom_pair[1]],
                "bodies": [body_pair[0], body_pair[1]],
                "count": 1,
                "min_distance": float(contact["distance"]),
                "max_normal_force": abs_normal_force,
            }
            continue

        current["count"] = int(current["count"]) + 1
        current["min_distance"] = float(min(float(current["min_distance"]), float(contact["distance"])))
        if abs_normal_force is not None:
            previous = current["max_normal_force"]
            if previous is None:
                current["max_normal_force"] = abs_normal_force
            else:
                current["max_normal_force"] = float(max(float(previous), abs_normal_force))
    return list(summary.values())


def _name_matches(value: str, needles: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(needle in lowered for needle in needles)


def _contact_labels(contact: dict[str, Any]) -> list[str]:
    return [
        str(contact["geom1"]),
        str(contact["geom2"]),
        str(contact["body1"]),
        str(contact["body2"]),
    ]


def _flags(contacts: list[dict[str, Any]]) -> dict[str, bool]:
    has_red_object_contact = False
    has_red_table_contact = False
    has_red_grasper_contact = False
    has_tip_contact = False
    has_obstacle_contact = False

    for contact in contacts:
        labels = _contact_labels(contact)
        has_red = any(_name_matches(label, ("red_object",)) for label in labels)
        has_table = any(_name_matches(label, ("tabletop", "table")) for label in labels)
        has_grasper = any(_name_matches(label, ("grasper", "finger", "feagine_grasper")) for label in labels)
        has_tip = any(_name_matches(label, ("tip", "feagine_grasper_tip")) for label in labels)
        has_obstacle = any(_name_matches(label, ("obstacle", "black_obstacle")) for label in labels)

        has_red_object_contact = has_red_object_contact or has_red
        has_red_table_contact = has_red_table_contact or (has_red and has_table)
        has_red_grasper_contact = has_red_grasper_contact or (has_red and has_grasper)
        has_tip_contact = has_tip_contact or has_tip
        has_obstacle_contact = has_obstacle_contact or has_obstacle

    return {
        "has_any_contact": bool(len(contacts) > 0),
        "has_red_object_contact": bool(has_red_object_contact),
        "has_red_table_contact": bool(has_red_table_contact),
        "has_red_grasper_contact": bool(has_red_grasper_contact),
        "has_tip_contact": bool(has_tip_contact),
        "has_obstacle_contact": bool(has_obstacle_contact),
    }


def diagnose_contacts(scene_path: str | Path, *, steps: int) -> dict[str, Any]:
    import mujoco

    resolved_scene = Path(scene_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(resolved_scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(max(0, steps))):
        mujoco.mj_step(model, data)

    red_body_id = _find_body_id(mujoco, model, "red_object")
    if red_body_id < 0:
        raise ValueError("Missing required body: red_object")

    contacts = [_contact_entry(mujoco, model, data, index) for index in range(int(data.ncon))]
    pair_summary = _pair_summary(contacts)
    flags = _flags(contacts)

    warnings: list[str] = []
    if not flags["has_red_table_contact"]:
        warnings.append("red_object has no tabletop contact after settling.")

    return {
        "scene": str(resolved_scene),
        "steps": int(max(0, steps)),
        "num_contacts": int(data.ncon),
        "contacts": contacts,
        "pair_summary": pair_summary,
        "flags": flags,
        "poses": {
            "red_object": _entity_pose(model, data, body_name="red_object"),
            "feagine_grasper_tip": _entity_pose(model, data, body_name="feagine_grasper_tip"),
            "tabletop": _entity_pose(model, data, geom_name="tabletop_geom"),
        },
        "warnings": warnings,
        "notes": [
            "This script only diagnoses contact pairs.",
            "It does not move the Feagine arm or implement grasping.",
        ],
    }


def write_contact_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        report = diagnose_contacts(scene_path=scene_path, steps=args.steps)
    except ValueError as exc:
        if "red_object" in str(exc):
            print("[FAIL] missing required body: red_object")
            return 2
        print(f"[FAIL] contact diagnosis failed: {exc}")
        return 1
    except Exception as exc:
        print(f"[FAIL] contact diagnosis failed: {exc}")
        return 1

    print(f"[INFO] loaded scene: {scene_path}")
    print(f"[INFO] steps={int(max(0, args.steps))}")
    print(f"[RESULT] num_contacts={report['num_contacts']}")
    for contact in report["contacts"]:
        if "red_object" in " ".join(_contact_labels(contact)).lower():
            print(
                f"[CONTACT] {contact['geom1']} <-> {contact['geom2']} "
                f"distance={contact['distance']} normal_force={contact['normal_force']}"
            )

    print(f"[FLAGS] has_red_object_contact={report['flags']['has_red_object_contact']}")
    print(f"[FLAGS] has_red_table_contact={report['flags']['has_red_table_contact']}")
    print(f"[FLAGS] has_red_grasper_contact={report['flags']['has_red_grasper_contact']}")

    for warning in report["warnings"]:
        print(f"[WARN] {warning}")

    try:
        saved_path = write_contact_report(report, output_path)
    except Exception as exc:
        print(f"[FAIL] failed to write diagnostics JSON: {exc}")
        return 3

    print(f"[OK] wrote {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
