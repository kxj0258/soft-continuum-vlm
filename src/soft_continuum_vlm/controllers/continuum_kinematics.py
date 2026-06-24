from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContinuumGeometry:
    section_count: int = 3
    section_length: float = 0.10
    max_abs_section_angle: float = 0.8
    damping: float = 1e-3


def section_angles_to_tip_delta(section_angles: Sequence[float], geometry: ContinuumGeometry) -> np.ndarray:
    """Approximate Feagine section-angle controls as a replaceable PCC proxy.

    This is not the final Feagine kinematic model. Each section has x/y bend
    controls; distal tip displacement is approximated by weighted cumulative
    bend. Later integration can replace this function with `pyfeagine_sim_core`
    PCC FK while keeping the controller interface stable.
    """

    angles = _validated_angles(section_angles, geometry)
    x_bends = angles[0::2]
    y_bends = angles[1::2]
    weights = np.linspace(float(geometry.section_count), 1.0, geometry.section_count) / geometry.section_count
    dx = float(geometry.section_length * np.dot(weights, x_bends))
    dy = float(geometry.section_length * np.dot(weights, y_bends))
    bend_energy = float(np.dot(x_bends, x_bends) + np.dot(y_bends, y_bends))
    straight_length = geometry.section_count * geometry.section_length
    dz = float(straight_length - 0.5 * geometry.section_length * bend_energy)
    return np.asarray([dx, dy, dz], dtype=np.float64)


def numeric_jacobian_tip_delta(
    section_angles: Sequence[float],
    geometry: ContinuumGeometry,
    eps: float = 1e-4,
) -> np.ndarray:
    angles = _validated_angles(section_angles, geometry)
    jacobian = np.zeros((3, angles.size), dtype=np.float64)
    for index in range(angles.size):
        plus = angles.copy()
        minus = angles.copy()
        plus[index] += eps
        minus[index] -= eps
        jacobian[:, index] = (
            section_angles_to_tip_delta(plus, geometry) - section_angles_to_tip_delta(minus, geometry)
        ) / (2.0 * eps)
    return jacobian


def damped_least_squares_step(
    current_section_angles: Sequence[float],
    target_tip_delta: Sequence[float],
    geometry: ContinuumGeometry,
    step_scale: float = 1.0,
) -> list[float]:
    current = _validated_angles(current_section_angles, geometry)
    target = np.asarray(target_tip_delta, dtype=np.float64).reshape(-1)
    if target.size != 3:
        raise ValueError("target_tip_delta must contain exactly 3 values.")
    jacobian = numeric_jacobian_tip_delta(current, geometry)
    lhs = jacobian.T @ jacobian + (geometry.damping**2) * np.eye(jacobian.shape[1])
    rhs = jacobian.T @ target
    try:
        delta_angles = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        delta_angles = np.linalg.pinv(lhs) @ rhs
    next_angles = current + float(step_scale) * delta_angles
    next_angles = np.clip(next_angles, -geometry.max_abs_section_angle, geometry.max_abs_section_angle)
    return [float(value) for value in next_angles]


def _validated_angles(section_angles: Sequence[float], geometry: ContinuumGeometry) -> np.ndarray:
    values = np.asarray(section_angles, dtype=np.float64).reshape(-1)
    expected = 2 * geometry.section_count
    if values.size != expected:
        raise ValueError(f"section_angles must contain {expected} values, got {values.size}.")
    return values
