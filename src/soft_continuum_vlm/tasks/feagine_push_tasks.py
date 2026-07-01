from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from soft_continuum_vlm.tasks.feagine_metaworld_task import FeagineMetaWorldTask


class FeaginePushTask(FeagineMetaWorldTask):
    def __init__(
        self,
        *,
        name: str,
        language: str,
        goal_position: Sequence[float],
        object_name: str = "push_object",
        approach_threshold: float = 0.04,
        success_threshold: float = 0.03,
        max_contact_force: float = 5.0,
        max_penetration: float = 0.01,
        contact_force_weight: float = 0.05,
        penetration_weight: float = 10.0,
        success_bonus: float = 2.0,
    ) -> None:
        self.name = name
        self.language = language
        self.object_name = object_name
        self._goal = self._point3(goal_position, "goal_position")
        self.approach_threshold = self._positive(approach_threshold, "approach_threshold")
        self.success_threshold = self._positive(success_threshold, "success_threshold")
        self.max_contact_force = self._positive(max_contact_force, "max_contact_force")
        self.max_penetration = self._positive(max_penetration, "max_penetration")
        self.contact_force_weight = self._nonnegative(
            contact_force_weight, "contact_force_weight"
        )
        self.penetration_weight = self._nonnegative(
            penetration_weight, "penetration_weight"
        )
        self.success_bonus = self._nonnegative(success_bonus, "success_bonus")
        self.phase = "approach"
        self._initial_object_position: np.ndarray | None = None
        self._target_contact_seen = False

    def reset_task(
        self,
        *,
        seed: int | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        del seed
        self.phase = "approach"
        self._target_contact_seen = False
        self._initial_object_position = (
            self.object_position(observation).copy()
            if observation is not None
            else None
        )

    def update_task_state(self, observation: Mapping[str, Any]) -> None:
        metrics = self.compute_metrics(observation)
        self._target_contact_seen = (
            self._target_contact_seen or metrics["target_contact_flag"] > 0.5
        )
        if self._success_from_metrics(metrics):
            self.phase = "complete"
        elif self.phase == "approach":
            if metrics["tip_object_distance"] <= self.approach_threshold:
                self.phase = "push"

    def compute_reward(self, observation: Mapping[str, Any]) -> float:
        metrics = self.compute_metrics(observation)
        reward = (
            -metrics["tip_object_distance"]
            -metrics["object_goal_distance"]
            -self.contact_force_weight * metrics["max_contact_force"]
            -self.penetration_weight * metrics["max_penetration"]
        )
        if self.compute_success(observation):
            reward += self.success_bonus
        return float(reward)

    def compute_success(self, observation: Mapping[str, Any]) -> bool:
        return self._success_from_metrics(self.compute_metrics(observation))

    def _success_from_metrics(self, metrics: Mapping[str, float]) -> bool:
        return bool(
            metrics["object_goal_distance"] < self.success_threshold
            and metrics["max_contact_force"] <= self.max_contact_force
            and metrics["max_penetration"] <= self.max_penetration
            and self._target_contact_seen
        )

    def get_goal(self) -> list[float]:
        return self._goal.astype(float).tolist()

    def get_object_state(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        objects = observation.get("objects", {})
        if not isinstance(objects, Mapping):
            raise ValueError("observation.objects must be a mapping.")
        state = objects.get(self.object_name)
        if not isinstance(state, Mapping):
            raise ValueError(f"observation.objects.{self.object_name} is required.")
        return state

    def get_task_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "language": self.language,
            "task_family": "push",
            "phase": self.phase,
            "object_name": self.object_name,
            "goal_position": self.get_goal(),
            "approach_threshold": float(self.approach_threshold),
            "success_threshold": float(self.success_threshold),
            "max_contact_force": float(self.max_contact_force),
            "max_penetration": float(self.max_penetration),
            "metric_keys": [
                "tip_object_distance",
                "object_goal_distance",
                "object_displacement",
                "max_contact_force",
                "max_penetration",
                "target_contact_flag",
                "target_contact_count",
                "target_contact_seen",
                "wrong_contact_count",
            ],
        }

    def get_task_context(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        object_position = self.object_position(observation)
        orientation_target = object_position if self.phase == "approach" else self._goal
        return {
            "phase": "approach",
            "task_phase": self.phase,
            "orientation_target_position": orientation_target.astype(float).tolist(),
            "goal_position": self.get_goal(),
        }

    def compute_metrics(self, observation: Mapping[str, Any]) -> dict[str, float]:
        tip = self.tip_position(observation)
        object_position = self.object_position(observation)
        contact = observation.get("contact", {})
        contact = contact if isinstance(contact, Mapping) else {}
        target_contacts, wrong_contacts = self._classify_contacts(contact)
        object_displacement = (
            float(np.linalg.norm(object_position - self._initial_object_position))
            if self._initial_object_position is not None
            else 0.0
        )
        return {
            "tip_object_distance": float(np.linalg.norm(tip - object_position)),
            "object_goal_distance": float(np.linalg.norm(object_position - self._goal)),
            "object_displacement": object_displacement,
            "max_contact_force": float(contact.get("max_force", 0.0)),
            "max_penetration": float(contact.get("max_penetration", 0.0)),
            "target_contact_flag": float(bool(target_contacts)),
            "target_contact_count": float(len(target_contacts)),
            "target_contact_seen": float(self._target_contact_seen),
            "wrong_contact_count": float(len(wrong_contacts)),
        }

    def object_position(self, observation: Mapping[str, Any]) -> np.ndarray:
        state = self.get_object_state(observation)
        pose = state.get("pose", {})
        if not isinstance(pose, Mapping) or "position" not in pose:
            raise ValueError(f"{self.object_name}.pose.position is required.")
        return self._point3(pose["position"], f"{self.object_name}.pose.position")

    def _classify_contacts(
        self,
        contact: Mapping[str, Any],
    ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        raw_contacts = contact.get("contacts", [])
        if not isinstance(raw_contacts, list):
            return [], []
        target: list[Mapping[str, Any]] = []
        wrong: list[Mapping[str, Any]] = []
        for item in raw_contacts:
            if not isinstance(item, Mapping):
                continue
            names = [
                str(item.get(key, ""))
                for key in ("geom1", "geom2", "body1", "body2")
            ]
            if any(self.object_name in name for name in names):
                target.append(item)
            else:
                wrong.append(item)
        return target, wrong

    @staticmethod
    def _point3(value: Any, label: str) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{label} must contain exactly three finite values.")
        return point

    @staticmethod
    def _positive(value: float, label: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{label} must be finite and positive.")
        return result

    @staticmethod
    def _nonnegative(value: float, label: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative.")
        return result


class FeaginePushLeftToRightTask(FeaginePushTask):
    def __init__(self, *, goal_position=(0.08, 0.0, 0.20), **kwargs: Any) -> None:
        super().__init__(
            name="feagine_push_left_to_right",
            language="Push the object from the left side to the right target.",
            goal_position=goal_position,
            **kwargs,
        )


class FeaginePushRightToLeftTask(FeaginePushTask):
    def __init__(self, *, goal_position=(-0.08, 0.0, 0.20), **kwargs: Any) -> None:
        super().__init__(
            name="feagine_push_right_to_left",
            language="Push the object from the right side to the left target.",
            goal_position=goal_position,
            **kwargs,
        )


class FeagineContactPushTask(FeaginePushTask):
    def __init__(self, *, goal_position=(0.0, 0.08, 0.20), **kwargs: Any) -> None:
        super().__init__(
            name="feagine_contact_push",
            language="Use controlled contact to push the object into the target region.",
            goal_position=goal_position,
            **kwargs,
        )


FEAGINE_PUSH_TASKS = {
    "feagine_push_left_to_right": FeaginePushLeftToRightTask,
    "feagine_push_right_to_left": FeaginePushRightToLeftTask,
    "feagine_contact_push": FeagineContactPushTask,
}


def make_feagine_push_task(name: str) -> FeaginePushTask:
    try:
        task_type = FEAGINE_PUSH_TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Feagine push task: {name}") from exc
    return task_type()
