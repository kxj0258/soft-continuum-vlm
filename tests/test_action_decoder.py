from __future__ import annotations

from soft_continuum_vlm.models.action_decoder import ACTION_KEYS, decode_feagine_action


def test_decode_feagine_action_uses_grasper_rotation_key() -> None:
    action = decode_feagine_action([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, -0.2], section_count=3)

    assert ACTION_KEYS == ("section_angles", "grip_command", "grasper_rotation")
    assert action == {
        "section_angles": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "grip_command": 1.0,
        "grasper_rotation": -0.2,
    }
