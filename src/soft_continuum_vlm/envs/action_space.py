from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


ACTION_DIM = 4
DEFAULT_DELTA_XYZ_SCALE = 0.01
GRIPPER_OPEN = -1.0
GRIPPER_CLOSED = 1.0


@dataclass(frozen=True)
class ScaledFeagineAction:
    """Task-space command produced from one normalized top-level action."""

    delta_xyz: np.ndarray
    gripper_control: float


class FeagineActionSpace:
    """Dependency-free description of the normalized MetaWorld-style action."""

    shape = (ACTION_DIM,)
    dtype = np.dtype(np.float32)

    def __init__(self) -> None:
        self.low = np.full(self.shape, -1.0, dtype=self.dtype)
        self.high = np.full(self.shape, 1.0, dtype=self.dtype)

    def contains(self, action: Sequence[float] | np.ndarray) -> bool:
        try:
            values = _as_action_vector(action)
        except (TypeError, ValueError):
            return False
        return bool(np.all(values >= self.low) and np.all(values <= self.high))

    def clip(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        values = _as_action_vector(action)
        return np.clip(values, self.low, self.high).astype(self.dtype, copy=False)


def scale_action(
    action: Sequence[float] | np.ndarray,
    *,
    delta_xyz_scale: float = DEFAULT_DELTA_XYZ_SCALE,
    clip: bool = True,
) -> ScaledFeagineAction:
    """Convert normalized action components into a task-space displacement."""

    if not np.isfinite(delta_xyz_scale) or delta_xyz_scale <= 0.0:
        raise ValueError("delta_xyz_scale must be a finite positive value.")

    action_space = FeagineActionSpace()
    values = action_space.clip(action) if clip else _as_action_vector(action)
    if not clip and not action_space.contains(values):
        raise ValueError("Action components must be within [-1, 1].")

    return ScaledFeagineAction(
        delta_xyz=(values[:3] * float(delta_xyz_scale)).astype(np.float32, copy=False),
        gripper_control=float(values[3]),
    )


def _as_action_vector(action: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Action must contain four numeric values.") from exc
    if values.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Action components must be finite.")
    return values
