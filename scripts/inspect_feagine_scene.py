from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.envs.feagine_mujoco_env import FeagineMujocoEnv
from soft_continuum_vlm.envs.mujoco_state import safe_mj_id2name
from soft_continuum_vlm.envs.scene_registry import SceneRegistry
from soft_continuum_vlm.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect real Feagine/MuJoCo scene names and observation schema.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--print-bodies", action="store_true")
    parser.add_argument("--print-geoms", action="store_true")
    parser.add_argument("--print-sites", action="store_true")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    env = FeagineMujocoEnv(config)
    try:
        observation = env.reset()
    except Exception as exc:
        print(
            "Unable to inspect Feagine scene because the real Feagine/MuJoCo runtime "
            f"could not be loaded or initialized: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        modules = env.ensure_runtime_available()
        mujoco = modules["mujoco"]
        model = env._model
        if model is None:
            raise RuntimeError("FeagineMujocoEnv did not expose a loaded MuJoCo model after reset().")
        counts = {
            "nbody": int(getattr(model, "nbody", 0)),
            "ngeom": int(getattr(model, "ngeom", 0)),
            "nsite": int(getattr(model, "nsite", 0)),
            "nu": int(getattr(model, "nu", 0)),
        }
        print(f"Model counts: {counts}")
        if args.print_bodies:
            print_names("Bodies", mujoco, model, mujoco.mjtObj.mjOBJ_BODY, counts["nbody"], args.limit)
        if args.print_geoms:
            print_names("Geoms", mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, counts["ngeom"], args.limit)
        if args.print_sites and hasattr(mujoco.mjtObj, "mjOBJ_SITE"):
            print_names("Sites", mujoco, model, mujoco.mjtObj.mjOBJ_SITE, counts["nsite"], args.limit)

        registry = SceneRegistry.from_config(config)
        resolved = registry.resolve(mujoco, model)
        for object_id, item in resolved.items():
            print(
                f"Object {object_id}: available={item.available} "
                f"body={item.body_name!r} geoms={list(item.geom_names)}"
            )

        for _ in range(max(0, int(args.steps))):
            robot_state = observation.get("robot_state", {})
            section_angles = list(robot_state.get("section_angles", [0.0] * 6))
            zero_action = {
                "section_angles": [0.0] * len(section_angles),
                "grip_command": 0.0,
                "grasper_rotation": 0.0,
            }
            observation, _reward, done, _info = env.step(zero_action)
            if done:
                break

        summary = build_summary(args, config, counts, resolved, observation)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
        print(f"Saved scene inspection summary to {output}")
        return 0
    finally:
        env.close()


def print_names(title: str, mujoco: Any, model: Any, obj_type: int, count: int, limit: int) -> None:
    names = [safe_mj_id2name(mujoco, model, obj_type, index) for index in range(max(0, count))]
    names = [name for name in names if name][:limit]
    print(f"{title} ({len(names)} shown):")
    for name in names:
        print(f"  {name}")


def build_summary(
    args: argparse.Namespace,
    config: dict[str, Any],
    counts: dict[str, int],
    resolved: Any,
    observation: Any,
) -> dict[str, Any]:
    robot_state = observation.get("robot_state", {})
    contact = observation.get("contact", {})
    qpos = robot_state.get("qpos", [])
    qvel = robot_state.get("qvel", [])
    section_angles = robot_state.get("section_angles", [])
    return {
        "config": str(args.config),
        "robot_preset": config.get("env", {}).get("robot_preset", ""),
        "model_counts": counts,
        "robot_state_keys": sorted(robot_state.keys()),
        "object_availability": {
            object_id: {
                "available": bool(item.available),
                "body_name": item.body_name,
                "geom_names": list(item.geom_names),
                "missing_reason": item.missing_reason,
            }
            for object_id, item in resolved.items()
        },
        "contact_summary": {
            "max_force": float(contact.get("max_force", 0.0)),
            "max_penetration": float(contact.get("max_penetration", 0.0)),
            "contact_count": len(contact.get("contacts", [])),
            "robot_contact_count": int(contact.get("robot_contact_count", 0)),
            "obstacle_contact_count": int(contact.get("obstacle_contact_count", 0)),
            "target_contact_count": int(contact.get("target_contact_count", 0)),
        },
        "qpos_shape": list(getattr(qpos, "shape", [len(qpos)])),
        "qvel_shape": list(getattr(qvel, "shape", [len(qvel)])),
        "section_count": int(len(section_angles) / 2) if section_angles else 0,
        "action_dimension": len(section_angles) + 2,
    }


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
