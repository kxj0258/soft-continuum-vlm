from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping

from soft_continuum_vlm.envs.mujoco_state import body_pose, safe_mj_id2name, safe_mj_name2id


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneObjectSpec:
    object_id: str
    body_names: list[str]
    geom_name_patterns: list[str]
    role: str
    color: str | None = None
    shape: str | None = None


@dataclass(frozen=True)
class ResolvedSceneObject:
    spec: SceneObjectSpec
    available: bool
    body_id: int | None = None
    body_name: str = ""
    geom_ids: tuple[int, ...] = ()
    geom_names: tuple[str, ...] = ()
    missing_reason: str = ""


class SceneRegistry:
    def __init__(self, objects: list[SceneObjectSpec]) -> None:
        self.objects = list(objects)
        self._resolved: dict[str, ResolvedSceneObject] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SceneRegistry":
        raw_scene = config.get("scene", {}) if isinstance(config, Mapping) else {}
        if not isinstance(raw_scene, Mapping):
            raw_scene = {}
        raw_objects = raw_scene.get("objects", [])
        specs: list[SceneObjectSpec] = []
        if isinstance(raw_objects, list):
            for item in raw_objects:
                if not isinstance(item, Mapping):
                    continue
                specs.append(
                    SceneObjectSpec(
                        object_id=str(item["object_id"]),
                        body_names=[str(value) for value in item.get("body_names", [])],
                        geom_name_patterns=[str(value) for value in item.get("geom_name_patterns", [])],
                        role=str(item.get("role", "target")),
                        color=str(item["color"]) if item.get("color") is not None else None,
                        shape=str(item["shape"]) if item.get("shape") is not None else None,
                    )
                )
        return cls(specs)

    def resolve(self, mujoco: Any, model: Any) -> dict[str, ResolvedSceneObject]:
        resolved = {spec.object_id: self._resolve_one(mujoco, model, spec) for spec in self.objects}
        self._resolved = resolved
        return dict(resolved)

    def object_pose(self, mujoco: Any, model: Any, data: Any, object_id: str) -> dict[str, Any]:
        resolved = self._resolved.get(object_id)
        if resolved is None:
            resolved = self.resolve(mujoco, model).get(object_id)
        if resolved is None or not resolved.available or resolved.body_id is None:
            return {"position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]}
        return body_pose(model, data, resolved.body_id)

    def build_objects_observation(self, mujoco: Any, model: Any, data: Any) -> dict[str, Any]:
        resolved_objects = self.resolve(mujoco, model)
        observation: dict[str, Any] = {}
        for object_id, resolved in resolved_objects.items():
            base: dict[str, Any] = {
                "available": resolved.available,
                "role": resolved.spec.role,
                "color": resolved.spec.color,
                "shape": resolved.spec.shape,
                "body_name": resolved.body_name,
                "geom_names": list(resolved.geom_names),
                "grasped": False,
            }
            if resolved.available:
                base["pose"] = self.object_pose(mujoco, model, data, object_id)
            else:
                base["missing_reason"] = resolved.missing_reason
            observation[object_id] = base
        return observation

    def _resolve_one(self, mujoco: Any, model: Any, spec: SceneObjectSpec) -> ResolvedSceneObject:
        body_obj_type = mujoco.mjtObj.mjOBJ_BODY
        geom_obj_type = mujoco.mjtObj.mjOBJ_GEOM
        body_id: int | None = None
        body_name = ""
        for candidate in spec.body_names:
            body_id = safe_mj_name2id(mujoco, model, body_obj_type, candidate)
            if body_id is not None:
                body_name = safe_mj_id2name(mujoco, model, body_obj_type, body_id)
                break

        matching_geom_ids = self._matching_geom_ids(mujoco, model, spec.geom_name_patterns)
        matching_geom_names = tuple(
            safe_mj_id2name(mujoco, model, geom_obj_type, geom_id) for geom_id in matching_geom_ids
        )

        if body_id is None and matching_geom_ids:
            body_id = self._geom_body_id(model, matching_geom_ids[0])
            if body_id is not None:
                body_name = safe_mj_id2name(mujoco, model, body_obj_type, body_id)

        if body_id is not None:
            geom_ids = matching_geom_ids or self._geom_ids_for_body(model, body_id)
            geom_names = tuple(
                safe_mj_id2name(mujoco, model, geom_obj_type, geom_id) for geom_id in geom_ids
            )
            return ResolvedSceneObject(
                spec=spec,
                available=True,
                body_id=body_id,
                body_name=body_name,
                geom_ids=tuple(geom_ids),
                geom_names=tuple(name for name in geom_names if name),
            )

        reason = (
            f"Could not resolve object_id={spec.object_id!r} from body_names={spec.body_names} "
            f"or geom_name_patterns={spec.geom_name_patterns}."
        )
        LOGGER.warning(
            "%s Available body names: %s. Available geom names: %s.",
            reason,
            self._name_summary(mujoco, model, body_obj_type, int(getattr(model, "nbody", 0))),
            self._name_summary(mujoco, model, geom_obj_type, int(getattr(model, "ngeom", 0))),
        )
        return ResolvedSceneObject(spec=spec, available=False, missing_reason=reason)

    def _matching_geom_ids(self, mujoco: Any, model: Any, patterns: list[str]) -> list[int]:
        lowered_patterns = [pattern.lower() for pattern in patterns]
        geom_count = int(getattr(model, "ngeom", len(getattr(model, "geom_bodyid", []))))
        matches: list[int] = []
        for geom_id in range(geom_count):
            name = safe_mj_id2name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            lowered_name = name.lower()
            if name and any(pattern in lowered_name for pattern in lowered_patterns):
                matches.append(geom_id)
        return matches

    @staticmethod
    def _geom_body_id(model: Any, geom_id: int) -> int | None:
        body_ids = getattr(model, "geom_bodyid", [])
        if geom_id < 0 or geom_id >= len(body_ids):
            return None
        return int(body_ids[geom_id])

    @staticmethod
    def _geom_ids_for_body(model: Any, body_id: int) -> list[int]:
        body_ids = getattr(model, "geom_bodyid", [])
        return [index for index, value in enumerate(body_ids) if int(value) == int(body_id)]

    @staticmethod
    def _name_summary(mujoco: Any, model: Any, obj_type: int, count: int, limit: int = 20) -> list[str]:
        names = [safe_mj_id2name(mujoco, model, obj_type, index) for index in range(max(0, count))]
        return [name for name in names if name][:limit]
