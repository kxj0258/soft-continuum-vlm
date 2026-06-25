from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a generated Feagine tabletop MuJoCo scene.")
    parser.add_argument("--scene", required=True, help="Path to generated scene XML.")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Optional viewer auto-close time.")
    return parser.parse_args()


def _exists(mujoco: Any, model: Any, object_type: Any, name: str) -> bool:
    try:
        return int(mujoco.mj_name2id(model, object_type, name)) >= 0
    except Exception:
        return False


def _viewer_is_running(viewer: Any) -> bool:
    is_running = getattr(viewer, "is_running", None)
    if callable(is_running):
        try:
            return bool(is_running())
        except Exception:
            return False
    return True


def _print_presence(mujoco: Any, model: Any) -> None:
    body_type = mujoco.mjtObj.mjOBJ_BODY
    geom_type = mujoco.mjtObj.mjOBJ_GEOM
    checks = {
        "red_object": _exists(mujoco, model, body_type, "red_object"),
        "red_object_geom": _exists(mujoco, model, geom_type, "red_object_geom"),
        "blue_object": _exists(mujoco, model, body_type, "blue_object"),
        "blue_object_geom": _exists(mujoco, model, geom_type, "blue_object_geom"),
        "black_obstacle": _exists(mujoco, model, body_type, "black_obstacle"),
        "black_obstacle_geom": _exists(mujoco, model, geom_type, "black_obstacle_geom"),
        "target_pad": _exists(mujoco, model, body_type, "target_pad"),
        "target_pad_geom": _exists(mujoco, model, geom_type, "target_pad_geom"),
        "feagine_grasper_tip": _exists(mujoco, model, body_type, "feagine_grasper_tip"),
    }
    for name, found in checks.items():
        print(f"[CHECK] {name}: {'FOUND' if found else 'MISSING'}")


def main() -> int:
    args = _parse_args()
    scene_path = Path(args.scene).expanduser().resolve()
    if not scene_path.exists():
        print(f"[FAIL] scene XML does not exist: {scene_path}")
        return 1

    try:
        import feagine_mujoco
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        print(f"[FAIL] MuJoCo/Feagine runtime import failed: {exc}")
        return 1

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        robot = feagine_mujoco.FeagineMjcfRobot(model, data, preset_id="a03_type_2")
        mujoco.mj_forward(model, data)
    except Exception as exc:
        print(f"[FAIL] scene load or Feagine wrapper creation failed: {exc}")
        return 1

    print(f"[OK] loaded scene: {scene_path}")
    print(f"[OK] created Feagine robot wrapper: {type(robot).__name__}")
    _print_presence(mujoco, model)

    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        print("[OK] MuJoCo viewer opened. Close the viewer window to exit.")
        start_time = time.monotonic()
        while _viewer_is_running(viewer):
            mujoco.mj_step(model, data)
            viewer.sync()
            if args.max_seconds > 0.0 and time.monotonic() - start_time >= args.max_seconds:
                break
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[FAIL] failed while opening or running MuJoCo viewer: {exc}")
        return 1
    finally:
        if viewer is not None and hasattr(viewer, "close"):
            try:
                viewer.close()
            except Exception:
                pass

    print("[OK] tabletop scene viewer finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
