from __future__ import annotations

from collections.abc import Sequence


# TODO(feagine-4d-action): Keep this decoder for low-level Feagine commands.
# Expected future top-level input is [dx, dy, dz, gripper_control], and the
# integration path is top-level policy -> FeagineActionAdapter -> decoded
# low-level command {section_angles, grip_command, grasper_rotation}.
ACTION_KEYS = ("section_angles", "grip_command", "grasper_rotation")


def decode_feagine_action(values: Sequence[float], *, section_count: int = 3) -> dict[str, object]:
    expected = 2 * section_count + 2
    if len(values) != expected:
        raise ValueError(f"Expected {expected} action values, got {len(values)}.")
    return {
        "section_angles": [float(value) for value in values[: 2 * section_count]],
        "grip_command": float(values[-2]),
        "grasper_rotation": float(values[-1]),
    }


decode_continuum_action = decode_feagine_action
