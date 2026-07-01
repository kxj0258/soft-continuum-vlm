import numpy as np

from soft_continuum_vlm.controllers.continuum_kinematics import ContinuumGeometry
from soft_continuum_vlm.workspace import (
    fit_workspace_ellipsoid,
    recommend_task_regions,
    sample_pcc_workspace,
)


def test_pcc_workspace_sampling_is_deterministic_and_bounded() -> None:
    geometry = ContinuumGeometry(max_abs_section_angle=0.8)
    first = sample_pcc_workspace(num_samples=32, seed=7, geometry=geometry)
    second = sample_pcc_workspace(num_samples=32, seed=7, geometry=geometry)

    np.testing.assert_allclose(first.section_angles, second.section_angles)
    np.testing.assert_allclose(first.tip_positions, second.tip_positions)
    assert first.section_angles.shape == (32, 6)
    assert first.tip_positions.shape == (32, 3)
    assert np.isfinite(first.tip_positions).all()
    assert np.max(np.abs(first.section_angles)) <= 0.8


def test_workspace_ellipsoid_reports_axes_and_reachable_ranges() -> None:
    points = np.asarray(
        [
            [-0.2, 0.0, 0.2],
            [0.2, 0.0, 0.2],
            [0.0, -0.1, 0.3],
            [0.0, 0.1, 0.3],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )

    fit = fit_workspace_ellipsoid(points)

    assert len(fit["center"]) == 3
    assert len(fit["principal_axes"]) == 3
    assert len(fit["semi_axis_lengths"]) == 3
    assert fit["left_right_reachable_range"] == [-0.2, 0.2]
    assert fit["front_back_reachable_range"] == [-0.1, 0.1]
    assert fit["height_range"] == [0.2, 0.5]


def test_recommended_regions_are_symmetric_and_inside_safe_workspace() -> None:
    rng = np.random.default_rng(4)
    points = rng.uniform([-0.3, -0.2, 0.15], [0.3, 0.2, 0.55], size=(200, 3))

    layout = recommend_task_regions(points, use_workspace_ratio=0.55)

    pick = layout["pick_region"]["center"]
    place = layout["place_region"]["center"]
    safe = layout["safe_ranges"]
    assert pick[0] < layout["workspace_center"][0] < place[0]
    assert safe["x"][0] <= pick[0] <= safe["x"][1]
    assert safe["x"][0] <= place[0] <= safe["x"][1]
    assert pick[2] == place[2]
    assert layout["feagine_base"]["orientation"] == "vertical_up"
