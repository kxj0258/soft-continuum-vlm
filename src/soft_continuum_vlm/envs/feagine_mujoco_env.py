from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module as default_import_module
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from soft_continuum_vlm.envs.base_env import Action, BaseRobotEnv, Observation
from soft_continuum_vlm.envs.mujoco_state import body_pose, contact_to_dict, safe_mj_id2name
from soft_continuum_vlm.envs.scene_registry import SceneRegistry


ImportModule = Callable[[str], Any]


@dataclass(frozen=True)
class FeagineMujocoConfig:
    robot_preset: str = "a03_type_2"
    asset_model_type: str = "mjcf"
    render_mode: str = "human"
    visual_preset: str = "debug_bright"
    viewer_camera: Mapping[str, Any] | None = None
    max_episode_steps: int = 200
    physics_steps_per_action: int = 4
    language: str = ""
    scene: Mapping[str, Any] | None = None


class FeagineMujocoEnv(BaseRobotEnv):
    """Thin Feagine MuJoCo wrapper using Feagine control methods as actions."""

    def __init__(
        self,
        config: Mapping[str, Any] | FeagineMujocoConfig | None = None,
        *,
        import_module: ImportModule = default_import_module,
    ) -> None:
        self.config = self._parse_config(config)
        self._import_module = import_module
        self._runtime_modules: dict[str, Any] = {}
        self._robot_asset_path: Path | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._robot: Any | None = None
        self._viewer: Any | None = None
        self._resolved_preset_id: str | None = None
        self._step_count = 0
        self._closed = False
        self._last_action: dict[str, Any] = {}
        self._last_section_angles: list[float] = []
        self._last_grip_command = 0.0
        self._last_grasper_rotation = 0.0
        self._scene_registry = SceneRegistry.from_config({"scene": self.config.scene or {}})
        self._observation = self._make_observation(self.config.language)

    def ensure_runtime_available(self) -> dict[str, Any]:
        if self._runtime_modules:
            return self._runtime_modules
        try:
            core = self._import_module("pyfeagine_sim_core")
            feagine_mujoco = self._import_module("feagine_mujoco")
            mujoco = self._import_module("mujoco")
        except ImportError as exc:
            raise ImportError(
                "Feagine MuJoCo runtime is unavailable. Expected imports "
                "'pyfeagine_sim_core', 'feagine_mujoco', and 'mujoco'. Activate "
                "the project environment and run scripts/setup_feagine_env.sh."
            ) from exc
        self._runtime_modules = {
            "pyfeagine_sim_core": core,
            "feagine_mujoco": feagine_mujoco,
            "mujoco": mujoco,
        }
        return self._runtime_modules

    def resolve_robot_asset_path(self) -> Path:
        if self._robot_asset_path is not None:
            return self._robot_asset_path
        mujoco = self.ensure_runtime_available()["feagine_mujoco"]
        robot_asset_path = getattr(mujoco, "robot_asset_path", None)
        if robot_asset_path is None:
            raise RuntimeError("feagine_mujoco.robot_asset_path is unavailable.")
        preset_id = self._resolved_preset_id or self._resolve_runtime_preset_id(mujoco)
        path = robot_asset_path(preset_id=preset_id, model_type=self.config.asset_model_type)
        self._robot_asset_path = Path(path).expanduser().resolve()
        return self._robot_asset_path

    def reset(self, *args: Any, **kwargs: Any) -> Observation:
        language = str(kwargs.get("language", self.config.language))
        self._step_count = 0
        self._closed = False
        self._last_action = {}
        self._last_section_angles = []
        self._last_grip_command = 0.0
        self._last_grasper_rotation = 0.0
        self._load_runtime()
        self._forward()
        self._ensure_viewer()
        self._observation = self._make_observation(language)
        return self.get_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._model is None or self._data is None or self._robot is None:
            self.reset()
        assert self._model is not None
        assert self._data is not None
        assert self._robot is not None

        applied_controls = self._apply_feagine_action(action)
        modules = self.ensure_runtime_available()
        mujoco = modules["mujoco"]
        for _ in range(self.config.physics_steps_per_action):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        self._last_action = dict(action)
        self._observation = self._make_observation(self._observation["language"])
        self._sync_viewer()
        done = self._step_count >= self.config.max_episode_steps
        info = {
            "runtime": "feagine_mujoco",
            "step_count": self._step_count,
            "last_action": self._last_action,
            "applied_controls": applied_controls,
            "resolved_preset_id": self._resolved_preset_id,
        }
        return self.get_observation(), 0.0, done, info

    def render(self) -> Any:
        self._sync_viewer()
        return self._observation["rgb"]

    def close(self) -> None:
        if self._viewer is not None and hasattr(self._viewer, "close"):
            self._viewer.close()
        self._viewer = None
        self._closed = True
        if "robot_state" in self._observation:
            self._observation["robot_state"] = self._read_robot_state()

    def get_observation(self) -> Observation:
        return dict(self._observation)

    def get_contact_info(self) -> Mapping[str, Any]:
        return dict(self._observation["contact"])

    def get_robot_state(self) -> Mapping[str, Any]:
        return dict(self._observation.get("robot_state", {}))

    def probe_tip_position_for_section_angles(
        self,
        section_angles: Sequence[float],
    ) -> list[float]:
        """Forward MuJoCo at a candidate section command without advancing time."""

        if self._model is None or self._data is None or self._robot is None:
            self.reset()
        assert self._model is not None
        assert self._data is not None
        assert self._robot is not None
        candidate = self._as_float_sequence(section_angles, "section_angles")
        expected = int(self._robot.section_count) * 2
        if len(candidate) != expected:
            raise ValueError(f"section_angles must contain {expected} values, got {len(candidate)}.")

        original_section_angles = self._read_section_angles()
        qpos = np.asarray(getattr(self._data, "qpos", [])).copy()
        qvel = np.asarray(getattr(self._data, "qvel", [])).copy()
        ctrl = np.asarray(getattr(self._data, "ctrl", [])).copy()
        time = float(getattr(self._data, "time", 0.0))
        modules = self.ensure_runtime_available()
        mujoco = modules["mujoco"]

        self._robot.drive_section_angles(candidate)
        for _ in range(self.config.physics_steps_per_action):
            mujoco.mj_step(self._model, self._data)
        tip = self._read_tip_pose()["position"]

        self._restore_probe_state(qpos=qpos, qvel=qvel, ctrl=ctrl, time=time)
        if len(original_section_angles) == expected:
            self._robot.drive_section_angles(original_section_angles)
            self._forward()
        return [float(value) for value in tip]

    def _restore_probe_state(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        time: float,
    ) -> None:
        assert self._data is not None
        self._assign_vector(self._data.qpos, qpos)
        self._assign_vector(self._data.qvel, qvel)
        if hasattr(self._data, "ctrl") and ctrl.size:
            self._assign_vector(self._data.ctrl, ctrl)
        if hasattr(self._data, "time"):
            self._data.time = float(time)
        self._forward()

    def _load_runtime(self) -> None:
        modules = self.ensure_runtime_available()
        feagine_mujoco = modules["feagine_mujoco"]
        mujoco = modules["mujoco"]
        self._resolved_preset_id = self._resolve_runtime_preset_id(feagine_mujoco)
        model = feagine_mujoco.load_mujoco_model(
            preset_id=self._resolved_preset_id,
            model_type=self.config.asset_model_type,
        )
        data = mujoco.MjData(model)
        if self.config.asset_model_type == "urdf":
            robot_class = feagine_mujoco.FeagineUrdfRobot
        else:
            robot_class = feagine_mujoco.FeagineMjcfRobot
        robot = robot_class(model, data, preset_id=self._resolved_preset_id)
        self._model = model
        self._data = data
        self._robot = robot
        self._apply_visual_preset()

    def _resolve_runtime_preset_id(self, feagine_mujoco: Any) -> str:
        preset_id = self.config.robot_preset
        try:
            feagine_mujoco.robot_asset_path(preset_id=preset_id, model_type=self.config.asset_model_type)
        except FileNotFoundError:
            if preset_id != "a03_type_2":
                raise
            fallback = "a03"
            feagine_mujoco.robot_asset_path(preset_id=fallback, model_type=self.config.asset_model_type)
            return fallback
        return preset_id

    def _apply_feagine_action(self, action: Action) -> list[str]:
        assert self._robot is not None
        # TODO(feagine-4d-action): Keep this env accepting low-level Feagine
        # commands. Expected future top-level input is [dx, dy, dz,
        # gripper_control], and the integration path is caller ->
        # FeagineActionAdapter -> {section_angles, grip_command,
        # grasper_rotation} -> FeagineMujocoEnv.step(...).
        supported = {
            "section_angles",
            "grip_command",
            "grasper_rotation",
            "joint_targets",
            "segment_joint_targets",
        }
        unknown = sorted(set(action) - supported)
        if unknown:
            raise ValueError(
                "Unsupported Feagine MuJoCo action field(s): "
                f"{unknown}. Use Feagine controls: section_angles, grip_command, "
                "grasper_rotation, joint_targets, or segment_joint_targets."
            )

        applied: list[str] = []
        if "section_angles" in action:
            section_angles = self._as_float_sequence(action["section_angles"], "section_angles")
            expected = int(self._robot.section_count) * 2
            if len(section_angles) != expected:
                raise ValueError(f"section_angles must contain {expected} values, got {len(section_angles)}.")
            self._robot.drive_section_angles(section_angles)
            self._last_section_angles = list(section_angles)
            applied.append("section_angles")
        if "grip_command" in action:
            grip_command = float(action["grip_command"])
            self._robot.set_grip_command(grip_command)
            self._last_grip_command = grip_command
            applied.append("grip_command")
        if "grasper_rotation" in action:
            grasper_rotation = float(action["grasper_rotation"])
            self._robot.drive_grasper_rotation(grasper_rotation)
            self._last_grasper_rotation = grasper_rotation
            applied.append("grasper_rotation")
        if "joint_targets" in action:
            joint_targets = {str(key): float(value) for key, value in dict(action["joint_targets"]).items()}
            self._robot.drive_joint_targets(joint_targets)
            applied.append("joint_targets")
        if "segment_joint_targets" in action:
            self._robot.set_segment_joint_targets(action["segment_joint_targets"])
            applied.append("segment_joint_targets")
        return applied

    def _forward(self) -> None:
        if self._model is None or self._data is None:
            return
        self.ensure_runtime_available()["mujoco"].mj_forward(self._model, self._data)

    def _ensure_viewer(self) -> None:
        if self.config.render_mode != "human" or self._viewer is not None:
            return
        if self._model is None or self._data is None:
            return
        mujoco = self.ensure_runtime_available()["mujoco"]
        viewer_module = getattr(mujoco, "viewer", None)
        if viewer_module is None:
            viewer_module = self._import_module("mujoco.viewer")
        launch_passive = getattr(viewer_module, "launch_passive", None)
        if launch_passive is None:
            raise RuntimeError("MuJoCo human render mode requires mujoco.viewer.launch_passive.")
        try:
            self._viewer = launch_passive(self._model, self._data)
        except Exception as exc:
            raise RuntimeError(
                "Failed to start MuJoCo human viewer. If this machine has no display "
                "or you are running a batch job, rerun the command with --headless "
                "or set render_mode: none."
            ) from exc
        self._apply_viewer_camera()

    def _sync_viewer(self) -> None:
        if self._viewer is not None and hasattr(self._viewer, "sync"):
            self._viewer.sync()

    def _apply_visual_preset(self) -> None:
        if self._model is None or self.config.visual_preset in {"", "none"}:
            return
        if self.config.visual_preset != "debug_bright":
            raise ValueError(
                f"Unsupported visual_preset={self.config.visual_preset!r}. "
                "Supported values are 'debug_bright' and 'none'."
            )
        headlight = getattr(getattr(self._model, "vis", None), "headlight", None)
        if headlight is None:
            return
        self._assign_vector(headlight.ambient, [0.6, 0.6, 0.6])
        self._assign_vector(headlight.diffuse, [0.8, 0.8, 0.8])
        self._assign_vector(headlight.specular, [0.3, 0.3, 0.3])

    def _apply_viewer_camera(self) -> None:
        if self._viewer is None or self.config.viewer_camera is None:
            return
        cam = getattr(self._viewer, "cam", None)
        if cam is None:
            return
        camera = dict(self.config.viewer_camera)
        if "lookat" in camera:
            lookat = self._as_float_sequence(camera["lookat"], "viewer_camera.lookat")
            if len(lookat) != 3:
                raise ValueError(f"viewer_camera.lookat must contain 3 values, got {len(lookat)}.")
            self._assign_vector(cam.lookat, lookat)
        for key in ("distance", "azimuth", "elevation"):
            if key in camera:
                setattr(cam, key, float(camera[key]))

    @staticmethod
    def _parse_config(
        config: Mapping[str, Any] | FeagineMujocoConfig | None,
    ) -> FeagineMujocoConfig:
        if isinstance(config, FeagineMujocoConfig):
            return config
        raw = dict(config or {})
        env_config = dict(raw.get("env", raw))
        return FeagineMujocoConfig(
            robot_preset=str(env_config.get("robot_preset", "a03_type_2")),
            asset_model_type=str(env_config.get("asset_model_type", "mjcf")),
            render_mode=str(env_config.get("render_mode", "human")),
            visual_preset=str(env_config.get("visual_preset", "debug_bright")),
            viewer_camera=dict(env_config["viewer_camera"]) if "viewer_camera" in env_config else None,
            max_episode_steps=int(env_config.get("max_episode_steps", 200)),
            physics_steps_per_action=int(env_config.get("physics_steps_per_action", 4)),
            language=str(env_config.get("language", "")),
            scene=dict(raw.get("scene", {})),
        )

    def _make_observation(self, language: str) -> Observation:
        contact = self._read_contact_info()
        proprioception = self._read_proprioception()
        robot_state = self._read_robot_state()
        objects = self._read_objects()
        return {
            "rgb": np.zeros((1, 1, 3), dtype=np.uint8),
            "depth": np.zeros((1, 1), dtype=np.float32),
            "proprioception": proprioception,
            "robot_state": robot_state,
            "objects": objects,
            "contact": contact,
            "language": language,
        }

    def _read_proprioception(self) -> np.ndarray:
        section_angles = self._read_section_angles()
        grip_command = self._read_grip_command()
        grasper_rotation = self._read_grasper_rotation()
        values = [
            *section_angles,
            grip_command,
            grasper_rotation,
        ]
        return np.asarray(values, dtype=np.float32)

    def _read_contact_info(self) -> dict[str, Any]:
        if self._model is None or self._data is None:
            return {
                "max_force": 0.0,
                "max_penetration": 0.0,
                "robot_contact_count": 0,
                "obstacle_contact_count": 0,
                "target_contact_count": 0,
                "contacts": [],
            }
        ncon = int(getattr(self._data, "ncon", 0))
        contacts: list[dict[str, Any]] = []
        max_force = 0.0
        max_penetration = 0.0
        modules = self.ensure_runtime_available()
        mujoco = modules["mujoco"]
        filters = self._contact_name_filters()
        for index in range(ncon):
            contact = contact_to_dict(mujoco, self._model, self._data, index, name_filters=filters)
            distance = float(contact["distance"])
            max_force = max(max_force, float(contact["force_norm"]))
            max_penetration = max(max_penetration, max(0.0, -distance))
            contacts.append(contact)
        return {
            "max_force": max_force,
            "max_penetration": max_penetration,
            "robot_contact_count": sum(1 for item in contacts if item["is_robot_contact"]),
            "obstacle_contact_count": sum(1 for item in contacts if item["is_obstacle_contact"]),
            "target_contact_count": sum(1 for item in contacts if item["is_target_contact"]),
            "contacts": contacts,
        }

    def _read_robot_state(self) -> dict[str, Any]:
        section_angles = self._read_section_angles()
        grip_command = self._read_grip_command()
        grasper_rotation = self._read_grasper_rotation()
        qpos = (
            np.asarray(getattr(self._data, "qpos", [])).copy()
            if self._data is not None
            else np.zeros(0)
        )
        qvel = (
            np.asarray(getattr(self._data, "qvel", [])).copy()
            if self._data is not None
            else np.zeros(0)
        )
        ctrl = (
            np.asarray(getattr(self._data, "ctrl", [])).copy()
            if self._data is not None
            else np.zeros(0)
        )
        tip_pose = self._read_tip_pose()
        return {
            "tip_pose": tip_pose,
            "section_angles": section_angles,
            "grip_command": grip_command,
            "grasper_rotation": grasper_rotation,
            "qpos": qpos,
            "qvel": qvel,
            "ctrl": ctrl,
            "step_count": self._step_count,
            "closed": self._closed,
        }

    def _read_section_angles(self) -> list[float]:
        if self._robot is not None:
            get_section_angles = getattr(self._robot, "get_section_angles", None)
            if callable(get_section_angles):
                try:
                    return [float(value) for value in get_section_angles()]
                except Exception:
                    pass
            section_angles = getattr(self._robot, "section_angles", None)
            if section_angles is not None:
                try:
                    return [float(value) for value in section_angles]
                except TypeError:
                    pass
        if self._last_section_angles:
            return list(self._last_section_angles)
        last_action_angles = self._last_action.get("section_angles")
        if last_action_angles is not None:
            return self._as_float_sequence(last_action_angles, "section_angles")
        return []

    def _read_grip_command(self) -> float:
        if self._robot is not None:
            value = getattr(self._robot, "grip_command", None)
            if value is not None:
                return float(value)
        if "grip_command" in self._last_action:
            return float(self._last_action["grip_command"])
        return float(self._last_grip_command)

    def _read_grasper_rotation(self) -> float:
        if self._robot is not None:
            value = getattr(self._robot, "grasper_rotation", None)
            if value is not None:
                return float(value)
        if "grasper_rotation" in self._last_action:
            return float(self._last_action["grasper_rotation"])
        return float(self._last_grasper_rotation)

    def _read_tip_pose(self) -> dict[str, Any]:
        if self._model is None or self._data is None:
            return self._unresolved_tip_pose()
        mujoco = self.ensure_runtime_available()["mujoco"]
        body_id = self._tip_body_id(mujoco, "feagine_grasper_tip")
        if body_id is None:
            return self._unresolved_tip_pose()
        position = np.asarray(self._data.xpos[body_id], dtype=np.float64).reshape(-1)[:3]
        orientation = np.asarray(self._data.xquat[body_id], dtype=np.float64).reshape(-1)[:4]
        return {
            "position": [float(value) for value in position],
            "orientation": [float(value) for value in orientation],
            "source": "body:feagine_grasper_tip",
            "body_name": "feagine_grasper_tip",
            "body_id": int(body_id),
        }

    def _tip_body_id(self, mujoco: Any, body_name: str) -> int | None:
        mjt_obj = getattr(mujoco, "mjtObj", None)
        mj_name2id = getattr(mujoco, "mj_name2id", None)
        if mjt_obj is None or mj_name2id is None:
            return None
        try:
            body_id = int(mj_name2id(self._model, mjt_obj.mjOBJ_BODY, body_name))
        except Exception:
            return None
        return body_id if body_id >= 0 else None

    def _read_robot_tip_pose(self) -> dict[str, list[float]] | None:
        if self._robot is None:
            return None
        for method_name in ("tip_pose", "get_tip_pose", "end_effector_pose", "get_end_effector_pose"):
            method = getattr(self._robot, method_name, None)
            if not callable(method):
                continue
            try:
                pose = method()
            except Exception:
                continue
            parsed = self._pose_object_to_dict(pose)
            if parsed is not None:
                return parsed
        return None

    def _find_named_site_pose(self, mujoco: Any) -> tuple[dict[str, list[float]], str] | None:
        mjt_obj = getattr(mujoco, "mjtObj", None)
        if mjt_obj is None:
            return None
        site_type = getattr(mjt_obj, "mjOBJ_SITE", None)
        if site_type is None:
            return None
        site_count = int(getattr(self._model, "nsite", 0))
        for site_id in range(site_count):
            name = safe_mj_id2name(mujoco, self._model, site_type, site_id)
            if not self._is_tip_like_name(name):
                continue
            position = np.asarray(self._data.site_xpos[site_id], dtype=np.float64).reshape(-1)[:3]
            if hasattr(self._data, "site_xquat"):
                orientation = np.asarray(self._data.site_xquat[site_id], dtype=np.float64).reshape(-1)[:4]
            else:
                matrix = np.asarray(self._data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
                orientation = self._mat3_to_quat(matrix)
            return {
                "position": [float(value) for value in position],
                "orientation": [float(value) for value in orientation],
            }, f"site:{name}"
        return None

    def _find_named_body_pose(self, mujoco: Any) -> tuple[dict[str, list[float]], str] | None:
        mjt_obj = getattr(mujoco, "mjtObj", None)
        if mjt_obj is None:
            return None
        body_count = int(getattr(self._model, "nbody", 0))
        for body_id in range(body_count):
            name = safe_mj_id2name(mujoco, self._model, mjt_obj.mjOBJ_BODY, body_id)
            if self._is_tip_like_name(name):
                return body_pose(self._model, self._data, body_id), f"body:{name}"
        return None

    def _read_objects(self) -> dict[str, Any]:
        if self._model is None or self._data is None or not self._scene_registry.objects:
            return {}
        mujoco = self.ensure_runtime_available()["mujoco"]
        return self._scene_registry.build_objects_observation(mujoco, self._model, self._data)

    def _contact_name_filters(self) -> dict[str, list[str]]:
        target_filters = ["target", "red", "blue", "cube", "cylinder"]
        obstacle_filters = ["obstacle", "black"]
        for spec in self._scene_registry.objects:
            patterns = [spec.object_id, *spec.body_names, *spec.geom_name_patterns]
            if spec.role == "obstacle":
                obstacle_filters.extend(patterns)
            elif spec.role == "target":
                target_filters.extend(patterns)
        return {
            "robot": ["robot", "arm", "section", "finger", "grasper", "gripper", "tip", "ee"],
            "obstacle": obstacle_filters,
            "target": target_filters,
        }

    def _contact_force_norm(self, mujoco: Any, contact_index: int) -> float:
        if self._model is None or self._data is None or not hasattr(mujoco, "mj_contactForce"):
            return 0.0
        values = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self._model, self._data, contact_index, values)
        return float(np.linalg.norm(values[:3]))

    @staticmethod
    def _pose_object_to_dict(pose: Any) -> dict[str, list[float]] | None:
        if isinstance(pose, Mapping):
            position = pose.get("position")
            orientation = pose.get("orientation", pose.get("quaternion"))
        else:
            position = getattr(pose, "position", None)
            orientation = getattr(pose, "orientation", getattr(pose, "quaternion", None))
        if position is None:
            return None
        orientation_values = [1.0, 0.0, 0.0, 0.0] if orientation is None else list(orientation)
        if len(orientation_values) == 0:
            orientation_values = [1.0, 0.0, 0.0, 0.0]
        return {
            "position": [float(value) for value in list(position)[:3]],
            "orientation": [float(value) for value in orientation_values[:4]],
        }

    @staticmethod
    def _zero_pose() -> dict[str, list[float]]:
        return {"position": [0.0, 0.0, 0.0], "orientation": [1.0, 0.0, 0.0, 0.0]}

    @staticmethod
    def _unresolved_tip_pose() -> dict[str, Any]:
        return {
            "position": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "source": "unresolved",
            "body_name": None,
            "body_id": None,
        }

    @staticmethod
    def _is_tip_like_name(name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in ("tip", "end_effector", "ee", "grasper"))

    @staticmethod
    def _mat3_to_quat(matrix: np.ndarray) -> np.ndarray:
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = float(np.sqrt(trace + 1.0) * 2.0)
            return np.asarray(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _as_float_sequence(value: Any, label: str) -> list[float]:
        if isinstance(value, np.ndarray):
            array = value.reshape(-1)
            return [float(item) for item in array]
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise TypeError(f"{label} must be a numeric sequence.")
        return [float(item) for item in value]

    @staticmethod
    def _assign_vector(target: Any, values: Sequence[float]) -> None:
        try:
            target[:] = values
        except TypeError:
            for index, value in enumerate(values):
                target[index] = value
