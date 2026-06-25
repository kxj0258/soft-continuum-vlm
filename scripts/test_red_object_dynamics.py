from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply an external force to red_object and report motion.")
    parser.add_argument("--scene", required=True, help="Path to the MuJoCo scene XML.")
    parser.add_argument(
        "--force",
        nargs=3,
        type=float,
        metavar=("FX", "FY", "FZ"),
        required=True,
        help="External Cartesian force applied to red_object.",
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps.")
    parser.add_argument("--output", required=True, help="Path to the output JSON diagnostics.")
    return parser.parse_args()


def _as_float_list(values) -> list[float]:
    return [float(value) for value in values]


def _detect_freejoint(model, body_id: int) -> bool:
    body_jntadr = int(model.body_jntadr[body_id])
    body_jntnum = int(model.body_jntnum[body_id])
    if body_jntnum <= 0:
        return False

    for joint_id in range(body_jntadr, body_jntadr + body_jntnum):
        if int(model.jnt_type[joint_id]) == 0:
            return True
    return False


def main() -> int:
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        print(f"[FAIL] MuJoCo is unavailable: {exc}")
        return 1

    args = _parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    red_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_object"))
    if red_body_id < 0:
        print("[FAIL] red_object body was not found.")
        return 1

    initial_position = np.array(data.xpos[red_body_id], copy=True)
    has_freejoint = _detect_freejoint(model, red_body_id)

    print(f"[INFO] red_object body_id={red_body_id}")
    print(f"[INFO] has_freejoint={has_freejoint}")

    if not has_freejoint:
        diagnostics = {
            "scene": str(scene_path),
            "force": _as_float_list(args.force),
            "steps": int(args.steps),
            "red_object": {
                "body_id": red_body_id,
                "initial_position": _as_float_list(initial_position),
                "final_position": _as_float_list(initial_position),
                "displacement": [0.0, 0.0, 0.0],
                "displacement_norm": 0.0,
                "has_freejoint": False,
            },
            "success": False,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(f"[RESULT] initial_position={diagnostics['red_object']['initial_position']}")
        print(f"[RESULT] final_position={diagnostics['red_object']['final_position']}")
        print("[RESULT] displacement_norm=0.0")
        print("[FAIL] red_object has no freejoint.")
        return 2

    force = np.array(args.force, dtype=float)
    for _ in range(int(args.steps)):
        data.xfrc_applied[red_body_id, :3] = force
        mujoco.mj_step(model, data)
    data.xfrc_applied[red_body_id, :] = 0.0
    mujoco.mj_forward(model, data)

    final_position = np.array(data.xpos[red_body_id], copy=True)
    displacement = final_position - initial_position
    displacement_norm = float(np.linalg.norm(displacement))

    diagnostics = {
        "scene": str(scene_path),
        "force": _as_float_list(force),
        "steps": int(args.steps),
        "red_object": {
            "body_id": red_body_id,
            "initial_position": _as_float_list(initial_position),
            "final_position": _as_float_list(final_position),
            "displacement": _as_float_list(displacement),
            "displacement_norm": displacement_norm,
            "has_freejoint": True,
        },
        "success": bool(has_freejoint and displacement_norm > 1e-4),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"[RESULT] initial_position={diagnostics['red_object']['initial_position']}")
    print(f"[RESULT] final_position={diagnostics['red_object']['final_position']}")
    print(f"[RESULT] displacement_norm={displacement_norm}")

    if displacement_norm <= 1e-4:
        print("[FAIL] red_object did not move enough.")
        return 3

    print("[OK] red_object is movable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
