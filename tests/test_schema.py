from __future__ import annotations

import numpy as np
import pytest

from soft_continuum_vlm.data.schema import (
    ACTION_KEYS,
    flatten_action,
    unflatten_action,
    validate_action,
    validate_observation,
)


def test_validate_action_rejects_gripper_rotation_with_fix_hint() -> None:
    with pytest.raises(ValueError, match="grasper_rotation"):
        validate_action(
            {
                "section_angles": [0.0] * 6,
                "grip_command": 0.0,
                "gripper_rotation": 0.0,
            }
        )


def test_action_schema_round_trip_uses_grasper_rotation() -> None:
    action = {
        "section_angles": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "grip_command": 1.0,
        "grasper_rotation": -0.25,
    }

    vector = flatten_action(action, section_count=3)
    restored = unflatten_action(vector, section_count=3)

    assert ACTION_KEYS == ("section_angles", "grip_command", "grasper_rotation")
    assert vector.shape == (8,)
    assert restored["section_angles"] == pytest.approx(action["section_angles"])
    assert restored["grip_command"] == pytest.approx(1.0)
    assert restored["grasper_rotation"] == pytest.approx(-0.25)


def test_validate_observation_reports_missing_required_field() -> None:
    with pytest.raises(ValueError, match="language"):
        validate_observation(
            {
                "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
                "depth": np.zeros((64, 64), dtype=np.float32),
                "proprioception": np.zeros(8, dtype=np.float32),
                "robot_state": {},
                "objects": {},
                "contact": {},
            }
        )
