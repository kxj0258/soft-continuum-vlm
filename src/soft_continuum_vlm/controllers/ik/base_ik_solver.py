from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class IkResult:
    """Structured solver result shared by global and differential IK."""

    success: bool
    converged: bool
    section_angles: list[float]
    achieved_tip_position: list[float] | None
    position_error_norm: float
    iterations: int
    status: str
    attempts: int = 1
    target_scale: float = 1.0

    @classmethod
    def failure(
        cls,
        current_section_angles: Sequence[float],
        status: str,
        position_error_norm: float,
        *,
        achieved_tip_position: Sequence[float] | None = None,
        iterations: int = 0,
    ) -> "IkResult":
        return cls(
            success=False,
            converged=False,
            section_angles=[float(value) for value in current_section_angles],
            achieved_tip_position=(
                [float(value) for value in achieved_tip_position]
                if achieved_tip_position is not None
                else None
            ),
            position_error_norm=float(position_error_norm),
            iterations=int(iterations),
            status=status,
        )


class IkSolver(ABC):
    """Pure IK boundary used by the future FeagineActionAdapter."""

    @abstractmethod
    def solve(
        self,
        target_tip_position: Sequence[float],
        current_section_angles: Sequence[float],
        *,
        current_tip_position: Sequence[float] | None = None,
    ) -> IkResult:
        """Return a safe section-angle command for one target position."""


def solve_with_retries(
    solver: IkSolver,
    target_tip_position: Sequence[float],
    current_section_angles: Sequence[float],
    *,
    current_tip_position: Sequence[float],
    retry_scales: Sequence[float] = (1.0, 0.5, 0.25),
) -> IkResult:
    """Retry progressively smaller Cartesian deltas, then hold current angles."""

    target = _point3(target_tip_position, "target_tip_position")
    current_tip = _point3(current_tip_position, "current_tip_position")
    scales = [float(value) for value in retry_scales]
    if not scales or any(not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in scales):
        raise ValueError("retry_scales must contain values within (0, 1].")

    for attempt, scale in enumerate(scales, start=1):
        scaled_target = current_tip + scale * (target - current_tip)
        result = solver.solve(
            scaled_target,
            current_section_angles,
            current_tip_position=current_tip,
        )
        if result.success:
            return replace(result, attempts=attempt, target_scale=scale)

    return replace(
        IkResult.failure(
            current_section_angles,
            "fallback_hold",
            float(np.linalg.norm(target - current_tip)),
            achieved_tip_position=current_tip,
        ),
        attempts=len(scales),
        target_scale=0.0,
    )


def point3(value: Sequence[float], label: str) -> np.ndarray:
    return _point3(value, label)


def section_angles(
    value: Sequence[float],
    *,
    expected: int,
    max_abs_section_angle: float,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected,) or not np.all(np.isfinite(array)):
        raise ValueError(f"section_angles must contain {expected} finite values.")
    if np.any(np.abs(array) > float(max_abs_section_angle) + 1e-12):
        raise ValueError("current section_angles exceed the configured limit.")
    return array


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain exactly three finite values.")
    return array
