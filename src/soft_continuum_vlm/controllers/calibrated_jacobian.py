from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class JacobianCalibration:
    preset: str
    model_type: str
    section_angle_length: int
    base_section_angles: list[float]
    base_tip_position: list[float]
    jacobian: list[list[float]]
    perturbation: float
    settle_steps: int
    tip_source: str
    active_columns: list[int]
    inactive_columns: list[int]
    column_norms: list[float]
    rank: int
    condition_number: float | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        array = np.asarray(self.jacobian, dtype=np.float64)
        expected_shape = (3, int(self.section_angle_length))
        if array.shape != expected_shape:
            raise ValueError(f"jacobian must have shape {expected_shape}, got {array.shape}.")

    def as_array(self) -> np.ndarray:
        return np.asarray(self.jacobian, dtype=np.float64)

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "JacobianCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
