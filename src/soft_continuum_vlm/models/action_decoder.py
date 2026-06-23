from __future__ import annotations

from collections.abc import Sequence


ACTION_KEYS = ("section_angles", "grip_command", "gripper_rotation")


def decode_feagine_action(values: Sequence[float], *, section_count: int = 3) -> dict[str, object]:
    expected = 2 * section_count + 2
    if len(values) != expected:
        raise ValueError(f"Expected {expected} action values, got {len(values)}.")
    return {
        "section_angles": [float(value) for value in values[: 2 * section_count]],
        "grip_command": float(values[-2]),
        "gripper_rotation": float(values[-1]),
    }


decode_continuum_action = decode_feagine_action
