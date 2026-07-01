from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.utils.paths import feagine_root  # noqa: E402


PREFERRED_ROBOT_PRESETS = ("a03_type_2", "a03")


def _try_import(module_name: str, *, required: bool) -> object | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if required:
            print(f"[ERROR] Could not import {module_name}: {exc}")
        else:
            print(f"[INFO] Optional {module_name} is not installed; MuJoCo-only setup is OK.")
        return None
    print(f"[OK] Imported {module_name}")
    return module


def _check_default_feagine_root() -> None:
    sibling_roots = [
        PROJECT_ROOT.parent / "feagine_simulation",
        PROJECT_ROOT.parent / "feagine-simulation",
    ]
    existing = [path for path in sibling_roots if path.exists()]
    if existing:
        print(f"[OK] Feagine sibling root exists: {existing[0]}")
    else:
        checked = ", ".join(str(path) for path in sibling_roots)
        print(f"[WARNING] Feagine sibling root is missing. Checked: {checked}")
    try:
        resolved = feagine_root()
    except FileNotFoundError as exc:
        print(f"[WARNING] {exc}")
    else:
        print(f"[OK] Resolved Feagine root: {resolved}")


def _print_asset_path(mujoco_module: object) -> Path | None:
    robot_asset_path = getattr(mujoco_module, "robot_asset_path", None)
    if robot_asset_path is None:
        print("[ERROR] feagine_mujoco.robot_asset_path is not available.")
        return None

    errors: list[str] = []
    for preset_id in PREFERRED_ROBOT_PRESETS:
        try:
            asset_path = Path(
                robot_asset_path(preset_id=preset_id, model_type="mjcf")
            ).expanduser().resolve()
        except Exception as exc:  # pragma: no cover - depends on external runtime
            errors.append(f"{preset_id}: {exc}")
            continue
        print(
            "[OK] feagine_mujoco.robot_asset_path("
            f"preset_id='{preset_id}', model_type='mjcf'): {asset_path}"
        )
        return asset_path

    print(
        "[ERROR] feagine_mujoco.robot_asset_path failed for presets "
        f"{list(PREFERRED_ROBOT_PRESETS)}. Errors: {'; '.join(errors)}"
    )
    return None


def _check_a03_type_2(asset_path: Path | None) -> None:
    if asset_path is None or not asset_path.exists():
        print("[WARNING] Cannot inspect a03_type_2 resources because asset path is unavailable.")
        return
    search_root = asset_path if asset_path.is_dir() else asset_path.parent
    if "a03_type_2" in search_root.parts:
        print(f"[OK] Found a03_type_2 preset asset path: {asset_path}")
    elif (search_root / "preset.yaml").exists() and "a03_type_2" in (search_root / "preset.yaml").read_text(
        encoding="utf-8"
    ):
        print(f"[OK] Found a03_type_2 preset metadata in: {search_root / 'preset.yaml'}")
    else:
        print(
            "[WARNING] Could not confirm a03_type_2 resources under "
            f"{search_root}. This may be fine if Feagine resolves presets internally."
        )


def main() -> int:
    print("[INFO] Verifying Feagine installation for soft-continuum-vlm")
    _check_default_feagine_root()

    core = _try_import("pyfeagine_sim_core", required=True)
    mujoco = _try_import("feagine_mujoco", required=True)
    _try_import("feagine_sapien", required=False)

    if core is None or mujoco is None:
        print(
            "[ERROR] Required Feagine MuJoCo imports failed. Activate the intended "
            "Python environment and run bash scripts/setup_feagine_env.sh."
        )
        return 1

    asset_path = _print_asset_path(mujoco)
    _check_a03_type_2(asset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
