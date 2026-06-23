from __future__ import annotations

from typing import Any

from soft_continuum_vlm.perception.scene_state import DetectedObject


class ObjectDetectorStub:
    """Deterministic no-weight detector used until VLM perception is integrated."""

    def detect(self, rgb: Any, *, language: str = "") -> list[DetectedObject]:
        _ = rgb, language
        return []
