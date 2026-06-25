from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the reachable Feagine tabletop scene.")
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--steps", type=int, default=200, help="Passive settling steps.")
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    return parser.parse_args()


def _to_float_list(values: Any, *, count: int | None = None) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if count is not None:
        array = array[:count]
    return [float(item) for item in array.tolist()]


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


def _find_joint_id(mujoco: Any, model: Any, name: str) -> int:
    try:
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    except Exception:
        return -1


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


def _contact_entry(mujoco: Any, model: Any, data: Any, index: int) -> dict[str, Any]:
    contact = data.contact[index]
    geom1_id = int(contact.geom1)
    geom2_id = int(contact.geom2)
    body1_id = int(model.geom_bodyid[geom1_id])
    body2_id = int(model.geom_bodyid[geom2_id])
    return {
        "geom1": _geom_name(model, geom1_id),
        "geom2": _geom_name(model, geom2_id),
        "body1": _body_name(model, body1_id),
        "body2": _body_name(model, body2_id),
        "distance": float(contact.dist),
        "position": _to_float_list(contact.pos, count=3),
    }


def _pair_matches(contact: dict[str, Any], left: str, right: str) -> bool:
    labels = {
        str(contact["geom1"]),
        str(contact["geom2"]),
        str(contact["body1"]),
        str(contact["body2"]),
    }
    return left in " ".join(labels).lower() and right in " ".join(labels).lower()


def _body_pose(model: Any, data: Any, body_id: int) -> dict[str, Any]:
    return {
        "position": _to_float_list(data.xpos[body_id], count=3),
        "orientation": _to_float_list(data.xquat[body_id], count=4),
    }


def validate_reachable_scene(scene_path: str | Path, *, steps: int) -> dict[str, Any]:
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
    red_freejoint_id = _find_joint_id(mujoco, model, "red_object_freejoint")
    if red_freejoint_id < 0:
        raise ValueError("Missing required freejoint: red_object_freejoint")
    pedestal_body_id = _find_body_id(mujoco, model, "red_pedestal")
    if pedestal_body_id < 0:
        raise ValueError("Missing required body: red_pedestal")
    tip_body_id = _find_body_id(mujoco, model, "feagine_grasper_tip")
    if tip_body_id < 0:
        raise ValueError("Missing required body: feagine_grasper_tip")

    contacts = [_contact_entry(mujoco, model, data, index) for index in range(int(data.ncon))]
    has_red_pedestal_contact = any(_pair_matches(contact, "red_object", "red_pedestal") for contact in contacts)
    has_red_table_contact = any(
        _pair_matches(contact, "red_object", "tabletop") or _pair_matches(contact, "red_object", "table")
        for contact in contacts
    )
    has_red_grasper_contact = any(
        _pair_matches(contact, "red_object", "grasper") or _pair_matches(contact, "red_object", "tip")
        for contact in contacts
    )

    red_pose = _body_pose(model, data, red_body_id)
    red_position = red_pose["position"]
    tip_position = _to_float_list(data.xpos[tip_body_id], count=3)
    tip_red_distance = float(np.linalg.norm(np.asarray(tip_position) - np.asarray(red_position)))
    success = bool(
        0.28 <= float(red_position[2]) <= 0.38
        and has_red_pedestal_contact
        and not has_red_table_contact
    )

    report = {
        "scene": str(resolved_scene),
        "steps": int(max(0, steps)),
        "red_object": {
            "position": red_position,
            "orientation": red_pose["orientation"],
            "has_freejoint": True,
        },
        "red_pedestal": {
            "position": _to_float_list(data.xpos[pedestal_body_id], count=3),
            "exists": True,
        },
        "feagine_grasper_tip": {"position": tip_position},
        "tip_red_distance": tip_red_distance,
        "contacts": contacts,
        "flags": {
            "has_red_pedestal_contact": has_red_pedestal_contact,
            "has_red_table_contact": has_red_table_contact,
            "has_red_grasper_contact": has_red_grasper_contact,
        },
        "success": success,
    }
    return report


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main() -> int:
    args = _parse_args()
    try:
        report = validate_reachable_scene(args.scene, steps=args.steps)
    except ValueError as exc:
        message = str(exc)
        if "red_object" in message and "freejoint" not in message:
            print("[FAIL] missing required body: red_object")
            return 2
        if "red_object_freejoint" in message:
            print("[FAIL] red_object is missing freejoint.")
            return 3
        if "red_pedestal" in message:
            print("[FAIL] missing required body: red_pedestal")
            return 4
        print(f"[FAIL] reachable scene validation failed: {exc}")
        return 1
    except Exception as exc:
        print(f"[FAIL] reachable scene validation failed: {exc}")
        return 1

    print(f"[INFO] loaded scene: {Path(args.scene).expanduser().resolve()}")
    print(f"[INFO] red_object has_freejoint={report['red_object']['has_freejoint']}")
    print(f"[INFO] red_pedestal exists={report['red_pedestal']['exists']}")
    print(f"[RESULT] red_object final position={report['red_object']['position']}")
    print(f"[RESULT] tip_red_distance={report['tip_red_distance']}")
    print(f"[FLAGS] has_red_pedestal_contact={report['flags']['has_red_pedestal_contact']}")
    print(f"[FLAGS] has_red_table_contact={report['flags']['has_red_table_contact']}")

    if not (0.28 <= float(report["red_object"]["position"][2]) <= 0.38):
        print("[FAIL] red_object final z out of range.")
        return 5
    if not report["flags"]["has_red_pedestal_contact"]:
        print("[FAIL] red_object has no red_pedestal contact.")
        return 6

    try:
        saved_path = write_report(report, args.output)
    except Exception as exc:
        print(f"[FAIL] failed to write diagnostics JSON: {exc}")
        return 7

    print(f"[OK] reachable scene validation passed")
    print(f"[OK] wrote {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
