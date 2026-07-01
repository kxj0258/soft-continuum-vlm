import numpy as np
import pytest

from soft_continuum_vlm.envs.action_space import (
    DEFAULT_DELTA_XYZ_SCALE,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    FeagineActionSpace,
    scale_action,
)


def test_feagine_action_space_has_fixed_metaworld_shape_and_bounds() -> None:
    action_space = FeagineActionSpace()

    assert action_space.shape == (4,)
    np.testing.assert_array_equal(action_space.low, np.full(4, -1.0, dtype=np.float32))
    np.testing.assert_array_equal(action_space.high, np.full(4, 1.0, dtype=np.float32))


def test_scale_action_maps_xyz_to_meters_without_changing_gripper_intent() -> None:
    scaled = scale_action([0.5, -1.0, 0.25, GRIPPER_CLOSED])

    np.testing.assert_allclose(
        scaled.delta_xyz,
        np.asarray([0.5, -1.0, 0.25], dtype=np.float32) * DEFAULT_DELTA_XYZ_SCALE,
    )
    assert scaled.gripper_control == pytest.approx(GRIPPER_CLOSED)


def test_scale_action_clips_normalized_input_by_default() -> None:
    scaled = scale_action([2.0, -3.0, 0.0, -4.0], delta_xyz_scale=0.005)

    np.testing.assert_allclose(scaled.delta_xyz, [0.005, -0.005, 0.0])
    assert scaled.gripper_control == pytest.approx(GRIPPER_OPEN)


@pytest.mark.parametrize(
    "action",
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, np.nan, 0.0, 0.0],
        [0.0, 0.0, np.inf, 0.0],
    ],
)
def test_scale_action_rejects_invalid_action_vectors(action: list[float]) -> None:
    with pytest.raises(ValueError):
        scale_action(action)


def test_action_space_contains_only_finite_normalized_4d_actions() -> None:
    action_space = FeagineActionSpace()

    assert action_space.contains([0.0, 1.0, -1.0, GRIPPER_OPEN])
    assert not action_space.contains([0.0, 1.01, 0.0, 0.0])
    assert not action_space.contains([0.0, 0.0, 0.0])
    assert not action_space.contains([0.0, 0.0, np.nan, 0.0])
