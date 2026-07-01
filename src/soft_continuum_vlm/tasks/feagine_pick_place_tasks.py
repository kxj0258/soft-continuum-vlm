from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from soft_continuum_vlm.tasks.feagine_metaworld_task import FeagineMetaWorldTask


PICK_PLACE_PHASES = (
    "approach",
    "align_grasper",
    "close_gripper",
    "lift",
    "transport",
    "align_place",
    "release",
    "retract",
    "complete",
)


class FeaginePickPlaceTask(FeagineMetaWorldTask):
    def __init__(
        self,
        *,
        name: str,
        language: str,
        goal_position: Sequence[float],
        object_name: str = "pick_object",
        place_principal_axis: Sequence[float] = (1.0, 0.0, 0.0),
        approach_threshold: float = 0.025,
        orientation_threshold: float = 0.12,
        lift_height: float = 0.06,
        transport_threshold: float = 0.035,
        place_threshold: float = 0.03,
        retract_distance: float = 0.05,
    ) -> None:
        self.name = name
        self.language = language
        self.object_name = object_name
        self._goal = self._point3(goal_position, "goal_position")
        self.place_principal_axis = self._axis(place_principal_axis, "place_principal_axis")
        self.approach_threshold = self._positive(approach_threshold, "approach_threshold")
        self.orientation_threshold = self._positive(
            orientation_threshold, "orientation_threshold"
        )
        self.lift_height = self._positive(lift_height, "lift_height")
        self.transport_threshold = self._positive(
            transport_threshold, "transport_threshold"
        )
        self.place_threshold = self._positive(place_threshold, "place_threshold")
        self.retract_distance = self._positive(retract_distance, "retract_distance")
        self.phase = "approach"
        self._initial_object_position: np.ndarray | None = None
        self._grasp_success = False
        self._lift_success = False
        self._place_success = False

    def reset_task(
        self,
        *,
        seed: int | None = None,
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        del seed
        self.phase = "approach"
        self._grasp_success = False
        self._lift_success = False
        self._place_success = False
        self._initial_object_position = (
            self.object_position(observation).copy()
            if observation is not None
            else None
        )

    def update_task_state(self, observation: Mapping[str, Any]) -> None:
        tip = self.tip_position(observation)
        object_position = self.object_position(observation)
        grasped = self.object_grasped(observation)
        grip_command = self.grip_command(observation)
        rotation = self.grasper_rotation(observation)
        object_axis = self.object_principal_axis(observation)

        self._grasp_success = self._grasp_success or grasped
        lifted_now = bool(
            self._grasp_success
            and object_position[2] >= self.initial_object_position[2] + self.lift_height
        )
        self._lift_success = self._lift_success or lifted_now
        placed_now = bool(
            np.linalg.norm(object_position - self._goal) < self.place_threshold
            and grip_command <= 0.2
            and not grasped
        )
        self._place_success = self._place_success or placed_now

        if self.phase == "approach":
            if np.linalg.norm(tip - object_position) <= self.approach_threshold:
                self.phase = "align_grasper"
        elif self.phase == "align_grasper":
            if self._rotation_error(rotation, self._axis_angle(object_axis)) <= self.orientation_threshold:
                self.phase = "close_gripper"
        elif self.phase == "close_gripper":
            if grasped:
                self.phase = "lift"
        elif self.phase == "lift":
            if lifted_now:
                self.phase = "transport"
        elif self.phase == "transport":
            preplace = self._goal + np.asarray([0.0, 0.0, self.lift_height])
            if np.linalg.norm(object_position - preplace) <= self.transport_threshold:
                self.phase = "align_place"
        elif self.phase == "align_place":
            place_rotation = self._axis_angle(self.place_principal_axis)
            if (
                np.linalg.norm(object_position - self._goal) <= self.place_threshold
                and self._rotation_error(rotation, place_rotation) <= self.orientation_threshold
            ):
                self.phase = "release"
        elif self.phase == "release":
            if placed_now:
                self.phase = "retract"
        elif self.phase == "retract":
            if placed_now and np.linalg.norm(tip - object_position) >= self.retract_distance:
                self.phase = "complete"

    def compute_reward(self, observation: Mapping[str, Any]) -> float:
        metrics = self.compute_metrics(observation)
        reward = -metrics["tip_object_distance"] - metrics["object_goal_distance"]
        reward += 0.5 * metrics["grasp_success"]
        reward += 1.0 * metrics["lift_success"]
        reward += 2.0 * metrics["place_success"]
        if self.compute_success(observation):
            reward += 3.0
        return float(reward)

    def compute_success(self, observation: Mapping[str, Any]) -> bool:
        return bool(
            self.phase == "complete"
            and self.current_place_success(observation)
            and self._grasp_success
            and self._lift_success
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
            "task_family": "pick_place",
            "phase": self.phase,
            "object_name": self.object_name,
            "goal_position": self.get_goal(),
            "pick_position": self.initial_object_position.astype(float).tolist(),
            "place_principal_axis": self.place_principal_axis.astype(float).tolist(),
            "lift_height": float(self.lift_height),
            "approach_threshold": float(self.approach_threshold),
            "place_threshold": float(self.place_threshold),
            "metric_keys": [
                "tip_object_distance",
                "object_goal_distance",
                "object_lift",
                "grasp_success",
                "lift_success",
                "place_success",
                "phase_index",
            ],
        }

    def get_task_context(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        object_position = self.object_position(observation).astype(float).tolist()
        object_axis = self.object_principal_axis(observation).astype(float).tolist()
        context: dict[str, Any] = {
            "phase": self.phase,
            "task_phase": self.phase,
            "goal_position": self.get_goal(),
        }
        if self.phase == "approach":
            context["orientation_target_position"] = object_position
        elif self.phase in {"align_grasper", "close_gripper"}:
            context["phase"] = "align_grasper" if self.phase == "align_grasper" else "grasp"
            context["object_principal_axis"] = object_axis
            context["orientation_target_position"] = object_position
        elif self.phase == "align_place":
            context["place_principal_axis"] = self.place_principal_axis.astype(float).tolist()
        return context

    def compute_metrics(self, observation: Mapping[str, Any]) -> dict[str, float]:
        tip = self.tip_position(observation)
        object_position = self.object_position(observation)
        return {
            "tip_object_distance": float(np.linalg.norm(tip - object_position)),
            "object_goal_distance": float(np.linalg.norm(object_position - self._goal)),
            "object_lift": float(object_position[2] - self.initial_object_position[2]),
            "grasp_success": float(self._grasp_success),
            "lift_success": float(self._lift_success),
            "place_success": float(self._place_success),
            "phase_index": float(PICK_PLACE_PHASES.index(self.phase)),
        }

    @property
    def initial_object_position(self) -> np.ndarray:
        if self._initial_object_position is None:
            return np.zeros(3, dtype=np.float64)
        return self._initial_object_position

    def object_position(self, observation: Mapping[str, Any]) -> np.ndarray:
        state = self.get_object_state(observation)
        pose = state.get("pose", {})
        if not isinstance(pose, Mapping):
            raise ValueError(f"{self.object_name}.pose is required.")
        return self._point3(pose.get("position"), f"{self.object_name}.pose.position")

    def object_grasped(self, observation: Mapping[str, Any]) -> bool:
        return bool(self.get_object_state(observation).get("grasped", False))

    def object_principal_axis(self, observation: Mapping[str, Any]) -> np.ndarray:
        state = self.get_object_state(observation)
        return self._axis(state.get("principal_axis", (1.0, 0.0, 0.0)), "principal_axis")

    @staticmethod
    def grip_command(observation: Mapping[str, Any]) -> float:
        state = observation.get("robot_state", {})
        if not isinstance(state, Mapping):
            raise ValueError("observation.robot_state must be a mapping.")
        value = float(state.get("grip_command", 0.0))
        if not np.isfinite(value):
            raise ValueError("grip_command must be finite.")
        return value

    @staticmethod
    def grasper_rotation(observation: Mapping[str, Any]) -> float:
        state = observation.get("robot_state", {})
        if not isinstance(state, Mapping):
            raise ValueError("observation.robot_state must be a mapping.")
        value = float(state.get("grasper_rotation", 0.0))
        if not np.isfinite(value):
            raise ValueError("grasper_rotation must be finite.")
        return value

    def current_place_success(self, observation: Mapping[str, Any]) -> bool:
        return bool(
            np.linalg.norm(self.object_position(observation) - self._goal) < self.place_threshold
            and self.grip_command(observation) <= 0.2
            and not self.object_grasped(observation)
        )

    @staticmethod
    def _axis_angle(axis: np.ndarray) -> float:
        return float(math.atan2(float(axis[1]), float(axis[0])))

    @staticmethod
    def _rotation_error(current: float, target: float) -> float:
        return abs(math.atan2(math.sin(current - target), math.cos(current - target)))

    @staticmethod
    def _point3(value: Any, label: str) -> np.ndarray:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{label} must contain exactly three finite values.")
        return point

    @classmethod
    def _axis(cls, value: Any, label: str) -> np.ndarray:
        axis = cls._point3(value, label)
        norm = float(np.linalg.norm(axis[:2]))
        if norm <= 1e-12:
            raise ValueError(f"{label} must have a nonzero xy component.")
        return axis / float(np.linalg.norm(axis))

    @staticmethod
    def _positive(value: float, label: str) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{label} must be finite and positive.")
        return result


class FeaginePickLeftPlaceRightTask(FeaginePickPlaceTask):
    def __init__(self, *, goal_position=(0.08, 0.0, 0.20), **kwargs: Any) -> None:
        super().__init__(
            name="feagine_pick_left_place_right",
            language="Pick the object from the left shelf and place it on the right shelf.",
            goal_position=goal_position,
            **kwargs,
        )


class FeaginePickRightPlaceLeftTask(FeaginePickPlaceTask):
    def __init__(self, *, goal_position=(-0.08, 0.0, 0.20), **kwargs: Any) -> None:
        super().__init__(
            name="feagine_pick_right_place_left",
            language="Pick the object from the right shelf and place it on the left shelf.",
            goal_position=goal_position,
            **kwargs,
        )


class FeaginePickShelfPlaceShelfTask(FeaginePickPlaceTask):
    def __init__(self, *, goal_position=(0.08, 0.04, 0.24), **kwargs: Any) -> None:
        super().__init__(
            name="feagine_pick_shelf_place_shelf",
            language="Pick the shelf object and place it at the target shelf pose.",
            goal_position=goal_position,
            **kwargs,
        )


FEAGINE_PICK_PLACE_TASKS = {
    "feagine_pick_left_place_right": FeaginePickLeftPlaceRightTask,
    "feagine_pick_right_place_left": FeaginePickRightPlaceLeftTask,
    "feagine_pick_shelf_place_shelf": FeaginePickShelfPlaceShelfTask,
}


def make_feagine_pick_place_task(name: str) -> FeaginePickPlaceTask:
    try:
        task_type = FEAGINE_PICK_PLACE_TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Feagine pick-place task: {name}") from exc
    return task_type()
