from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from soft_continuum_vlm.data.schema import DATASET_REQUIRED_KEYS


def save_demo_npz(path: str | Path, **arrays: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = [key for key in DATASET_REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Demo dataset is missing required array(s): {missing}")
    np.savez_compressed(output, **arrays)
    return output


def load_demo_npz(path: str | Path) -> dict[str, np.ndarray]:
    loaded = np.load(Path(path), allow_pickle=True)
    data = {key: loaded[key] for key in loaded.files}
    missing = [key for key in DATASET_REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Demo dataset is missing required array(s): {missing}")
    return data
