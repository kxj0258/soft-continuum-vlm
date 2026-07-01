from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from soft_continuum_vlm.controllers.continuum_kinematics import (
    ContinuumGeometry,
    section_angles_to_tip_delta,
)


@dataclass(frozen=True)
class WorkspaceSamples:
    """Deterministic PCC commands and their finite, reachable tip positions."""

    section_angles: np.ndarray
    tip_positions: np.ndarray
    rejected_count: int
    backend: str = "approximate_pcc"


def sample_pcc_workspace(
    *,
    num_samples: int,
    seed: int = 0,
    geometry: ContinuumGeometry | None = None,
    base_position: Sequence[float] = (0.0, 0.0, 0.0),
) -> WorkspaceSamples:
    """Sample the replaceable PCC FK model without requiring MuJoCo."""

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    resolved_geometry = geometry or ContinuumGeometry()
    base = _point3(base_position, "base_position")
    rng = np.random.default_rng(seed)
    angles = rng.uniform(
        -resolved_geometry.max_abs_section_angle,
        resolved_geometry.max_abs_section_angle,
        size=(num_samples, 2 * resolved_geometry.section_count),
    )
    angles[0] = 0.0

    positions = np.asarray(
        [
            base + section_angles_to_tip_delta(command, resolved_geometry)
            for command in angles
        ],
        dtype=np.float64,
    )
    finite = np.all(np.isfinite(positions), axis=1)
    above_base = positions[:, 2] >= base[2]
    valid = finite & above_base
    return WorkspaceSamples(
        section_angles=angles[valid],
        tip_positions=positions[valid],
        rejected_count=int(np.count_nonzero(~valid)),
    )


def fit_workspace_ellipsoid(points: Sequence[Sequence[float]] | np.ndarray) -> dict[str, Any]:
    """Fit a PCA-aligned bounding ellipsoid to reachable points."""

    array = _points(points)
    center = np.mean(array, axis=0)
    centered = array - center
    _, _, vectors_t = np.linalg.svd(centered, full_matrices=False)
    principal_axes = vectors_t
    projected = centered @ principal_axes.T
    semi_axis_lengths = np.max(np.abs(projected), axis=0)
    order = np.argsort(semi_axis_lengths)[::-1]
    principal_axes = principal_axes[order]
    semi_axis_lengths = semi_axis_lengths[order]

    return {
        "center": center.astype(float).tolist(),
        "principal_axes": principal_axes.astype(float).tolist(),
        "semi_axis_lengths": semi_axis_lengths.astype(float).tolist(),
        "major_axis": float(semi_axis_lengths[0]),
        "minor_axis": float(semi_axis_lengths[-1]),
        "height_range": _range(array[:, 2]),
        "left_right_reachable_range": _range(array[:, 0]),
        "front_back_reachable_range": _range(array[:, 1]),
    }


def recommend_task_regions(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    use_workspace_ratio: float = 0.55,
) -> dict[str, Any]:
    """Place symmetric pick/place regions inside robust workspace quantiles."""

    if not 0.0 < use_workspace_ratio <= 1.0:
        raise ValueError("use_workspace_ratio must be within (0, 1].")
    array = _points(points)
    lower = np.quantile(array, 0.05, axis=0)
    upper = np.quantile(array, 0.95, axis=0)
    center = (lower + upper) / 2.0
    safe_half_extent = (upper - lower) * use_workspace_ratio / 2.0
    safe_lower = center - safe_half_extent
    safe_upper = center + safe_half_extent
    lateral_offset = safe_half_extent[0] * 0.5
    region_half_extent = np.maximum(safe_half_extent * 0.18, 1e-6)
    task_height = center[2]

    def region(side: str, x: float) -> dict[str, Any]:
        return {
            "side": side,
            "center": [float(x), float(center[1]), float(task_height)],
            "half_extent": region_half_extent.astype(float).tolist(),
        }

    return {
        "feagine_base": {
            "position": [0.0, 0.0, 0.0],
            "orientation": "vertical_up",
        },
        "workspace_center": center.astype(float).tolist(),
        "use_workspace_ratio": float(use_workspace_ratio),
        "safe_ranges": {
            "x": [float(safe_lower[0]), float(safe_upper[0])],
            "y": [float(safe_lower[1]), float(safe_upper[1])],
            "z": [float(safe_lower[2]), float(safe_upper[2])],
        },
        "pick_region": region("left", center[0] - lateral_offset),
        "place_region": region("right", center[0] + lateral_offset),
    }


def write_workspace_outputs(
    samples: WorkspaceSamples,
    output_dir: str | Path,
    *,
    use_workspace_ratio: float = 0.55,
) -> dict[str, Path]:
    """Write the standard point cloud, JSON report, and PNG visualization."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    points_path = output / "feagine_workspace_points.npy"
    report_path = output / "feagine_workspace.json"
    image_path = output / "feagine_workspace.png"

    ellipsoid = fit_workspace_ellipsoid(samples.tip_positions)
    layout = recommend_task_regions(
        samples.tip_positions,
        use_workspace_ratio=use_workspace_ratio,
    )
    report = {
        "schema_version": 1,
        "backend": samples.backend,
        "sample_count": int(samples.tip_positions.shape[0]),
        "rejected_count": int(samples.rejected_count),
        "ellipsoid": ellipsoid,
        "recommended_layout": layout,
        "notes": [
            "The approximate_pcc backend is deterministic and does not perform MuJoCo collision checks.",
            "Confirm the recommended layout with the Feagine simulation backend before scene freeze.",
        ],
    }

    np.save(points_path, samples.tip_positions)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot_workspace(samples.tip_positions, layout, image_path)
    return {"points": points_path, "report": report_path, "image": image_path}


def _plot_workspace(points: np.ndarray, layout: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, alpha=0.35)
    for key, color in (("pick_region", "tab:blue"), ("place_region", "tab:orange")):
        center = layout[key]["center"]
        axis.scatter(*center, s=70, color=color, label=key)
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_zlabel("world z (m)")
    axis.set_title("Feagine PCC reachable workspace")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _points(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] < 4:
        raise ValueError("points must have shape (N, 3) with N >= 4.")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values.")
    return array


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain exactly three finite values.")
    return array


def _range(values: np.ndarray) -> list[float]:
    return [float(np.min(values)), float(np.max(values))]
