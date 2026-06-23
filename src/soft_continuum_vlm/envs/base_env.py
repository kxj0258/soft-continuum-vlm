from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


Observation = dict[str, Any]
Action = Mapping[str, Any]


class BaseRobotEnv(ABC):
    """Abstract interface shared by simulated and real robot environments."""

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Observation:
        """Reset the environment and return the first observation."""

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Apply one action and return observation, reward, done, and info."""

    @abstractmethod
    def render(self) -> Any:
        """Render the current state or return the current RGB observation."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Return the latest structured observation."""

    @abstractmethod
    def get_contact_info(self) -> Mapping[str, Any]:
        """Return contact force, penetration, and related safety metadata."""

    @abstractmethod
    def get_robot_state(self) -> Mapping[str, Any]:
        """Return proprioceptive robot state used by controllers and models."""
