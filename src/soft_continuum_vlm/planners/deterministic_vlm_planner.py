from __future__ import annotations

from typing import Any, Mapping


COLOR_TO_OBJECT = {
    "红": "red_object",
    "red": "red_object",
    "蓝": "blue_object",
    "blue": "blue_object",
    "绿": "green_object",
    "green": "green_object",
    "黄": "yellow_object",
    "yellow": "yellow_object",
}

COLOR_TO_OBSTACLE = {
    "黑": "black_obstacle",
    "black": "black_obstacle",
}


class DeterministicVLMPlanner:
    """Rule-based, offline planner stub that mimics structured VLM output."""

    def __init__(self, *, default_force_limit: float = 1.0, gentle_force_limit: float = 0.5) -> None:
        self.default_force_limit = default_force_limit
        self.gentle_force_limit = gentle_force_limit

    def plan(self, language: str, observation: Mapping[str, Any], task_name: str) -> dict[str, Any]:
        text = str(language).lower()
        target_object = self._target_object(text, observation, task_name)
        avoid_objects = self._avoid_objects(text, observation)
        gentle = any(token in text for token in ("轻轻", "gentle", "gently", "softly", "小心"))
        avoid_collision = bool(avoid_objects) or any(token in text for token in ("绕过", "不要碰", "avoid", "collision"))
        requires_rotation = any(token in text for token in ("旋转", "rotate", "rotation")) or task_name == "rotate_and_place"
        pushing = any(token in text for token in ("推动", "推", "push")) or task_name == "contact_push"
        placing = any(token in text for token in ("放置", "place")) or task_name == "rotate_and_place"
        contact_force_limit = self.gentle_force_limit if gentle else self.default_force_limit
        subgoals = self._subgoals(
            target_object=target_object,
            avoid_objects=avoid_objects,
            pushing=pushing,
            placing=placing,
            requires_rotation=requires_rotation,
        )
        return {
            "target_object": target_object,
            "avoid_objects": avoid_objects,
            "approach_side": self._approach_side(text),
            "grasp_mode": "gentle" if gentle else "normal",
            "contact_force_limit": float(contact_force_limit),
            "requires_rotation": requires_rotation,
            "requires_push": pushing,
            "subgoals": subgoals,
            "language_constraints": {
                "avoid_collision": avoid_collision,
                "gentle_contact": gentle,
                "requires_rotation": requires_rotation,
            },
        }

    def _target_object(self, text: str, observation: Mapping[str, Any], task_name: str) -> str:
        for token, object_name in COLOR_TO_OBJECT.items():
            if token in text:
                return object_name
        objects = observation.get("objects", {})
        if isinstance(objects, Mapping):
            for preferred in ("red_object", "target_object", "push_object", "grasped_object"):
                if preferred in objects:
                    return preferred
        return {
            "pick_red_object": "red_object",
            "obstacle_avoid_pick": "target_object",
            "contact_push": "push_object",
            "rotate_and_place": "grasped_object",
        }.get(task_name, "target_object")

    def _avoid_objects(self, text: str, observation: Mapping[str, Any]) -> list[str]:
        objects = observation.get("objects", {})
        candidates: list[str] = []
        for token, object_name in COLOR_TO_OBSTACLE.items():
            if token in text:
                candidates.append(object_name)
        if any(token in text for token in ("绕过", "avoid", "不要碰")) and isinstance(objects, Mapping):
            for name in objects:
                if "obstacle" in str(name) and name not in candidates:
                    candidates.append(str(name))
        return candidates

    @staticmethod
    def _approach_side(text: str) -> str:
        if "left" in text or "左" in text:
            return "left"
        if "right" in text or "右" in text:
            return "right"
        if "front" in text or "前" in text:
            return "front"
        return "any"

    @staticmethod
    def _subgoals(
        *,
        target_object: str,
        avoid_objects: list[str],
        pushing: bool,
        placing: bool,
        requires_rotation: bool,
    ) -> list[dict[str, Any]]:
        subgoals: list[dict[str, Any]] = [{"phase": "approach", "target": target_object}]
        if avoid_objects:
            subgoals.append({"phase": "avoid", "target": avoid_objects[0]})
        if pushing:
            subgoals.append({"phase": "push", "target": target_object})
        else:
            subgoals.append({"phase": "grasp", "target": target_object})
        if requires_rotation:
            subgoals.append({"phase": "rotate", "target": target_object})
        subgoals.append({"phase": "transport", "target": target_object})
        if placing:
            subgoals.append({"phase": "release", "target": target_object})
        return subgoals
