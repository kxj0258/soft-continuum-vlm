from __future__ import annotations

import argparse
import json
import pathlib


KEYWORDS = [
    "tip",
    "ee",
    "end",
    "end_effector",
    "grasper",
    "gripper",
    "finger",
    "left",
    "right",
    "jaw",
    "palm",
    "wrist",
    "base",
    "section",
    "segment",
    "soft",
    "continuum",
    "arm",
    "target",
    "object",
    "obstacle",
    "red",
    "blue",
    "cube",
    "cylinder",
]

TIP_SCORES = {
    "tip": 100,
    "end": 80,
    "end_effector": 80,
    "ee": 80,
    "grasper": 60,
    "gripper": 60,
    "finger": 50,
    "jaw": 50,
    "palm": 40,
    "wrist": 40,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Feagine MuJoCo model names.")
    parser.add_argument("--preset", default="a03_type_2")
    parser.add_argument("--model-type", default="mjcf")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output", default="outputs/diagnostics/mujoco_model_names.json")
    return parser.parse_args()


def _load_runtime() -> tuple[object, object] | None:
    try:
        import feagine_mujoco
        import mujoco
    except ImportError as exc:
        print(f"[FAIL] import failed: {exc}")
        return None
    return feagine_mujoco, mujoco


def _load_model(feagine_mujoco: object, mujoco: object, preset: str, model_type: str) -> tuple[object, object, str] | None:
    try:
        asset_path = feagine_mujoco.robot_asset_path(preset_id=preset, model_type=model_type)
        model = mujoco.MjModel.from_xml_path(str(asset_path))
    except TypeError as exc:
        print(f"[WARN] robot_asset_path(preset_id=..., model_type=...) incompatible: {exc}")
        try:
            model = feagine_mujoco.load_mujoco_model(preset_id=preset, model_type=model_type)
        except Exception as load_exc:
            print(f"[FAIL] load_mujoco_model failed: {load_exc}")
            return None
        asset_path = ""
    except Exception as exc:
        print(f"[FAIL] model load failed: {exc}")
        return None

    try:
        data = mujoco.MjData(model)
    except Exception as exc:
        print(f"[FAIL] mujoco.MjData(model) failed: {exc}")
        return None
    return model, data, str(asset_path)


def _counts(model: object) -> dict[str, int]:
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nsite": int(model.nsite),
        "njnt": int(model.njnt),
        "nsensor": int(getattr(model, "nsensor", 0)),
    }


def _names(mujoco: object, model: object, obj_type: object, count: int, limit: int) -> list[str]:
    names: list[str] = []
    for index in range(min(count, limit)):
        try:
            name = mujoco.mj_id2name(model, obj_type, index)
        except Exception:
            name = None
        names.append("" if name is None else str(name))
    return names


def _collect_names(mujoco: object, model: object, counts: dict[str, int], limit: int) -> dict[str, list[str]]:
    return {
        "body": _names(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, counts["nbody"], limit),
        "geom": _names(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, counts["ngeom"], limit),
        "site": _names(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, counts["nsite"], limit),
        "joint": _names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, counts["njnt"], limit),
        "actuator": _names(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, counts["nu"], limit),
        "sensor": _names(mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, counts["nsensor"], limit),
    }


def _keyword_matches(names: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    matches: dict[str, dict[str, list[str]]] = {}
    for keyword in KEYWORDS:
        lowered_keyword = keyword.lower()
        matches[keyword] = {}
        for kind, kind_names in names.items():
            matches[keyword][kind] = [
                name for name in kind_names if name and lowered_keyword in name.lower()
            ]
    return matches


def _tip_candidates(names: dict[str, list[str]]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for kind in ("body", "geom"):
        for name in names[kind]:
            lowered_name = name.lower()
            matched = [keyword for keyword in TIP_SCORES if keyword in lowered_name]
            if not matched:
                continue
            score = sum(TIP_SCORES[keyword] for keyword in matched)
            candidates.append(
                {
                    "name": name,
                    "kind": kind,
                    "score": score,
                    "matched_keywords": matched,
                }
            )
    candidates.sort(key=lambda item: int(item["score"]), reverse=True)
    return candidates


def _write_json(output: str, payload: dict[str, object]) -> bool:
    path = pathlib.Path(output)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[FAIL] could not write {output}: {exc}")
        return False
    print(f"[OK] wrote {output}")
    return True


def _print_summary(preset: str, counts: dict[str, int], tip_candidates: list[dict[str, object]], actuator_names: list[str]) -> None:
    print(f"[OK] loaded preset: {preset}")
    print(
        "[INFO] "
        f"nq={counts['nq']}, nv={counts['nv']}, nu={counts['nu']}, "
        f"nbody={counts['nbody']}, ngeom={counts['ngeom']}, nsite={counts['nsite']}"
    )
    if counts["nsite"] == 0:
        print("[INFO] site count is 0; tip source will likely need body/geom fallback.")
    if tip_candidates:
        print("[INFO] top tip candidates:")
        for index, candidate in enumerate(tip_candidates[:10], start=1):
            print(
                f"  {index}. kind={candidate['kind']}, "
                f"name={candidate['name']}, score={candidate['score']}"
            )
    else:
        print("[WARN] no obvious tip candidate found from names; inspect body/geom names manually.")
    print("[INFO] actuator names:")
    for name in actuator_names:
        print(f"  {name}")


def main() -> int:
    args = _parse_args()
    runtime = _load_runtime()
    if runtime is None:
        return 1
    feagine_mujoco, mujoco = runtime

    loaded = _load_model(feagine_mujoco, mujoco, args.preset, args.model_type)
    if loaded is None:
        return 2
    model, data, asset_path = loaded
    _ = data

    counts = _counts(model)
    names = _collect_names(mujoco, model, counts, args.limit)
    keyword_matches = _keyword_matches(names)
    tip_candidates = _tip_candidates(names)
    control_candidates = {
        "actuator_names": names["actuator"],
        "joint_names": names["joint"],
        "nu": counts["nu"],
        "nv": counts["nv"],
        "nq": counts["nq"],
    }
    payload = {
        "preset": args.preset,
        "model_type": args.model_type,
        "asset_path": asset_path,
        "counts": counts,
        "names": names,
        "keyword_matches": keyword_matches,
        "tip_candidates_ranked": tip_candidates,
        "control_candidates": control_candidates,
    }

    _print_summary(args.preset, counts, tip_candidates, names["actuator"])
    if not _write_json(args.output, payload):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
