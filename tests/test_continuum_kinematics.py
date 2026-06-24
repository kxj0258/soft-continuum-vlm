from __future__ import annotations

import numpy as np
import pytest

from soft_continuum_vlm.controllers.continuum_kinematics import (
    ContinuumGeometry,
    damped_least_squares_step,
    numeric_jacobian_tip_delta,
    section_angles_to_tip_delta,
)


def test_section_angles_to_tip_delta_maps_positive_x_bend_to_positive_x() -> None:
    geometry = ContinuumGeometry(section_count=3, section_length=0.1)

    delta = section_angles_to_tip_delta([0.1, 0.0, 0.1, 0.0, 0.1, 0.0], geometry)

    assert delta.shape == (3,)
    assert delta[0] > 0.0
    assert abs(delta[1]) < 1e-9
    assert delta[2] < geometry.section_count * geometry.section_length


def test_numeric_jacobian_tip_delta_is_finite_and_has_expected_shape() -> None:
    geometry = ContinuumGeometry(section_count=2, section_length=0.1)

    jacobian = numeric_jacobian_tip_delta([0.0, 0.0, 0.1, -0.1], geometry)

    assert jacobian.shape == (3, 4)
    assert np.isfinite(jacobian).all()
    assert abs(jacobian[0, 0]) > 0.0
    assert abs(jacobian[1, 1]) > 0.0


def test_damped_least_squares_step_moves_toward_positive_x_target() -> None:
    geometry = ContinuumGeometry(section_count=3, section_length=0.1, max_abs_section_angle=0.8)

    next_angles = damped_least_squares_step([0.0] * 6, [0.04, 0.0, 0.0], geometry)

    assert len(next_angles) == 6
    assert max(next_angles[0::2]) > 0.0
    assert all(abs(value) <= geometry.max_abs_section_angle for value in next_angles)


def test_section_angle_length_is_validated() -> None:
    with pytest.raises(ValueError, match="section_angles"):
        section_angles_to_tip_delta([0.0, 0.0], ContinuumGeometry(section_count=3))
