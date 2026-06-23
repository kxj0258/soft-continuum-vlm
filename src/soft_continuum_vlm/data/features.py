from __future__ import annotations

import hashlib

import numpy as np


def encode_language_stub(language: str | list[str], dim: int = 64) -> np.ndarray:
    if dim <= 0:
        raise ValueError("Language feature dim must be positive.")
    if isinstance(language, list):
        return np.stack([encode_language_stub(item, dim=dim) for item in language]).astype(np.float32)
    vector = np.zeros(dim, dtype=np.float32)
    text = str(language)
    for char in text:
        digest = hashlib.blake2b(char.encode("utf-8"), digest_size=4).digest()
        index = int.from_bytes(digest, "little") % dim
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def build_morphology_vector(
    section_count: int,
    has_grasper_rotation: bool = True,
    has_grip: bool = True,
) -> np.ndarray:
    return np.asarray(
        [
            float(section_count),
            float(2 * section_count),
            1.0 if has_grasper_rotation else 0.0,
            1.0 if has_grip else 0.0,
        ],
        dtype=np.float32,
    )
