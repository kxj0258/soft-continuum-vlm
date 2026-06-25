from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Feagine tabletop MuJoCo scene XML.")
    parser.add_argument("--output", required=True, help="Path for the generated scene XML.")
    parser.add_argument("--variant", choices=("default", "reachable"), default="default")
    return parser.parse_args()


def _robot_asset_path() -> Path:
    try:
        import feagine_mujoco
    except ImportError as exc:
        raise ImportError(
            "feagine_mujoco is unavailable. Activate the Feagine environment before generating the scene."
        ) from exc

    path = feagine_mujoco.robot_asset_path(preset_id="a03_type_2", model_type="mjcf")
    return Path(path).expanduser().resolve()


def _ensure_asset(root: ET.Element) -> ET.Element:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    return asset


def _resolve_asset_dir_value(source_dir: Path, value: str | None) -> str:
    if not value:
        return source_dir.as_posix()
    path = Path(value)
    if path.is_absolute():
        return path.as_posix()
    return (source_dir / path).resolve().as_posix()


def _ensure_compiler_paths(root: ET.Element, source_dir: Path) -> ET.Element:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", _resolve_asset_dir_value(source_dir, compiler.get("meshdir")))
    compiler.set("texturedir", _resolve_asset_dir_value(source_dir, compiler.get("texturedir")))
    return compiler


def _append_tabletop_geom(worldbody: ET.Element) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "tabletop_geom",
            "type": "box",
            "pos": "0.22 0.0 -0.025",
            "size": "0.38 0.28 0.025",
            "rgba": "0.82 0.82 0.78 1",
        },
    )


def _append_red_object(worldbody: ET.Element, *, position: str) -> None:
    red_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "red_object", "pos": position},
    )
    ET.SubElement(red_body, "freejoint", {"name": "red_object_freejoint"})
    ET.SubElement(
        red_body,
        "geom",
        {
            "name": "red_object_geom",
            "type": "box",
            "size": "0.025 0.025 0.025",
            "mass": "0.05",
            "friction": "0.8 0.1 0.1",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "0.9 0.05 0.03 1",
        },
    )


def _append_red_pedestal(worldbody: ET.Element) -> None:
    pedestal_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "red_pedestal", "pos": "0.34 0.0 0.1535"},
    )
    ET.SubElement(
        pedestal_body,
        "geom",
        {
            "name": "red_pedestal_geom",
            "type": "box",
            "size": "0.08 0.08 0.1535",
            "rgba": "0.70 0.70 0.72 1",
            "contype": "1",
            "conaffinity": "1",
        },
    )


def _append_support_objects(worldbody: ET.Element, *, variant: str) -> None:
    obstacle_position = "0.34 0.0 0.045"
    if variant == "reachable":
        obstacle_position = "0.18 -0.14 0.045"

    ET.SubElement(
        worldbody,
        "body",
        {"name": "blue_object", "pos": "0.22 0.08 0.045"},
    ).append(
        ET.Element(
            "geom",
            {
                "name": "blue_object_geom",
                "type": "cylinder",
                "size": "0.022 0.035",
                "rgba": "0.05 0.18 0.9 1",
            },
        )
    )
    ET.SubElement(
        worldbody,
        "body",
        {"name": "black_obstacle", "pos": obstacle_position},
    ).append(
        ET.Element(
            "geom",
            {
                "name": "black_obstacle_geom",
                "type": "box",
                "size": "0.035 0.035 0.045",
                "rgba": "0.02 0.02 0.02 1",
            },
        )
    )


def _append_tabletop_scene(worldbody: ET.Element, *, variant: str) -> None:
    _append_tabletop_geom(worldbody)
    if variant == "reachable":
        _append_red_pedestal(worldbody)
        _append_red_object(worldbody, position="0.34 0.0 0.332")
    else:
        _append_red_object(worldbody, position="0.24 -0.08 0.03")
    _append_support_objects(worldbody, variant=variant)
    ET.SubElement(
        worldbody,
        "body",
        {"name": "target_pad", "pos": "0.12 0.11 0.002"},
    ).append(
        ET.Element(
            "geom",
            {
                "name": "target_pad_geom",
                "type": "cylinder",
                "size": "0.055 0.003",
                "rgba": "0.0 0.9 0.25 0.35",
            },
        )
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "tabletop_camera",
            "pos": "0.62 -0.55 0.42",
            "xyaxes": "0.66 0.75 0.0 -0.32 0.28 0.91",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "tabletop_key_light",
            "pos": "0.2 -0.25 0.8",
            "dir": "-0.1 0.15 -1",
            "diffuse": "0.9 0.9 0.85",
            "specular": "0.3 0.3 0.3",
        },
    )


def generate_tabletop_scene(output_path: str | Path, *, variant: str = "default") -> Path:
    source_path = _robot_asset_path()
    tree = ET.parse(source_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Original Feagine MJCF has no <worldbody>: {source_path}")

    _ensure_asset(root)
    _ensure_compiler_paths(root, source_path.parent)
    _append_tabletop_scene(worldbody, variant=variant)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def main() -> int:
    args = _parse_args()
    try:
        output_path = generate_tabletop_scene(args.output, variant=args.variant)
    except Exception as exc:
        print(f"[FAIL] tabletop scene generation failed: {exc}")
        return 1
    print(f"[OK] generated tabletop scene: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
