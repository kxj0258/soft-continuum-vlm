from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def zero_feagine_action(section_count: int) -> dict[str, object]:
    return {
        "section_angles": [0.0] * (2 * section_count),
        "grip_command": 0.0,
        "grasper_rotation": 0.0,
    }


@dataclass
class PccIkController:
    """Placeholder for a piecewise-constant-curvature IK controller."""

    section_count: int = 3

    def compute_action(
        self,
        target_state: Mapping[str, Any],
        robot_state: Mapping[str, Any],
    ) -> dict[str, object]:
        _ = target_state, robot_state
        return zero_feagine_action(self.section_count)
