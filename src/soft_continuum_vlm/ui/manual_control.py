from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ManualControlConfig:
    section_angle_length: int = 6
    section_step: float = 0.03
    grasper_rotation_step: float = 0.05
    max_abs_section_angle: float = 0.8
    min_grip_command: float = 0.0
    max_grip_command: float = 1.0


@dataclass
class ManualControlState:
    section_angles: list[float] = field(default_factory=list)
    grip_command: float = 0.0
    grasper_rotation: float = 0.0
    paused: bool = False
    quit_requested: bool = False


class ManualFeagineController:
    _SECTION_KEY_DELTAS = {
        "1": (0, 1.0),
        "q": (0, -1.0),
        "2": (2, 1.0),
        "w": (2, -1.0),
        "3": (4, 1.0),
        "e": (4, -1.0),
        "a": (1, 1.0),
        "z": (1, -1.0),
        "s": (3, 1.0),
        "x": (3, -1.0),
        "d": (5, 1.0),
        "c": (5, -1.0),
    }

    def __init__(self, config: ManualControlConfig | None = None) -> None:
        self.config = config or ManualControlConfig()
        self.state = ManualControlState(
            section_angles=[0.0] * int(self.config.section_angle_length),
        )

    def reset(self, section_angles: list[float] | None = None) -> None:
        angles = self._coerce_section_angles(section_angles)
        self.state = ManualControlState(
            section_angles=angles,
            grip_command=float(self.config.min_grip_command),
            grasper_rotation=0.0,
            paused=False,
            quit_requested=False,
        )

    def handle_key(self, key: str) -> dict[str, Any]:
        normalized = "space" if key == " " else key.lower()

        if normalized in self._SECTION_KEY_DELTAS:
            index, direction = self._SECTION_KEY_DELTAS[normalized]
            step = float(self.config.section_step) * float(direction)
            self.state.section_angles[index] = self._clip_section_angle(
                self.state.section_angles[index] + step
            )
        elif normalized == "o":
            self.state.grip_command = float(self.config.min_grip_command)
        elif normalized == "p":
            self.state.grip_command = float(self.config.max_grip_command)
        elif normalized == "r":
            self.state.grasper_rotation += float(self.config.grasper_rotation_step)
        elif normalized == "f":
            self.state.grasper_rotation -= float(self.config.grasper_rotation_step)
        elif normalized == "space":
            self.state.paused = not self.state.paused
        elif normalized == "0":
            self.reset()
        elif normalized == "esc":
            self.state.quit_requested = True

        return {
            "action": self.action(),
            "paused": bool(self.state.paused),
            "quit_requested": bool(self.state.quit_requested),
            "status_text": self.status_text(),
        }

    def action(self) -> dict[str, Any]:
        return {
            "section_angles": [float(value) for value in self.state.section_angles],
            "grip_command": float(self.state.grip_command),
            "grasper_rotation": float(self.state.grasper_rotation),
        }

    def status_text(self) -> str:
        angles = ", ".join(f"{value:.3f}" for value in self.state.section_angles)
        return (
            f"section_angles=[{angles}] "
            f"grip={self.state.grip_command:.3f} "
            f"rotation={self.state.grasper_rotation:.3f} "
            f"paused={self.state.paused}"
        )

    def _coerce_section_angles(self, section_angles: list[float] | None) -> list[float]:
        length = int(self.config.section_angle_length)
        if section_angles is None:
            return [0.0] * length
        result = [0.0] * length
        for index, value in enumerate(section_angles[:length]):
            result[index] = self._clip_section_angle(float(value))
        return result

    def _clip_section_angle(self, value: float) -> float:
        limit = float(self.config.max_abs_section_angle)
        return max(-limit, min(limit, float(value)))
