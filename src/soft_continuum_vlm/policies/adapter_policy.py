from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from soft_continuum_vlm.controllers.safety_projector import SafetyLimits, SafetyProjector
from soft_continuum_vlm.data.features import build_morphology_vector, encode_language_stub
from soft_continuum_vlm.data.schema import flatten_contact, flatten_proprioception
from soft_continuum_vlm.models.action_decoder import decode_feagine_action
from soft_continuum_vlm.models.soft_embodiment_adapter import SoftEmbodimentAdapter


class AdapterPolicy:
    baseline_name = "adapter"

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        section_count: int = 3,
        safety_projector: SafetyProjector | None = None,
        device: str = "cpu",
    ) -> None:
        import torch

        self.torch = torch
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found: {self.checkpoint_path}")
        self.section_count = section_count
        self.device = torch.device(device)
        checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)
        dims = dict(checkpoint_data.get("input_dims", {}))
        hidden_dim = int(checkpoint_data.get("config", {}).get("hidden_dim", 64))
        self.input_dims = {
            "vision_language_dim": int(dims.get("vision_language_dim", 64)),
            "proprioception_dim": int(dims.get("proprioception_dim", 2 * section_count + 2)),
            "contact_dim": int(dims.get("contact_dim", 3)),
            "morphology_dim": int(dims.get("morphology_dim", 4)),
            "action_dim": int(dims.get("action_dim", 2 * section_count + 2)),
        }
        self.model = SoftEmbodimentAdapter(
            vision_language_dim=self.input_dims["vision_language_dim"],
            proprioception_dim=self.input_dims["proprioception_dim"],
            contact_dim=self.input_dims["contact_dim"],
            morphology_dim=self.input_dims["morphology_dim"],
            hidden_dim=hidden_dim,
            action_dim=self.input_dims["action_dim"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        self.model.eval()
        self.safety_projector = safety_projector or SafetyProjector(
            SafetyLimits(
                max_abs_section_angle=0.8,
                max_gripper_rotation=1.0,
                max_contact_force=1.0,
                max_penetration=0.01,
            )
        )
        self.task_name = ""
        self.language = ""

    def reset(self, task_name: str, language: str | None = None) -> None:
        self.task_name = task_name
        self.language = language or ""

    def act(self, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        language = str(observation.get("language", self.language))
        language_feature = self._fit(
            encode_language_stub(language, dim=self.input_dims["vision_language_dim"]),
            self.input_dims["vision_language_dim"],
        )
        proprioception = self._fit(flatten_proprioception(observation), self.input_dims["proprioception_dim"])
        contact = observation.get("contact", {})
        contact_state = self._fit(flatten_contact(contact if isinstance(contact, Mapping) else {}), self.input_dims["contact_dim"])
        morphology = self._fit(build_morphology_vector(self.section_count), self.input_dims["morphology_dim"])
        with torch.no_grad():
            prediction = self.model(
                vision_language_feature=torch.as_tensor(language_feature, dtype=torch.float32, device=self.device).unsqueeze(0),
                proprioception=torch.as_tensor(proprioception, dtype=torch.float32, device=self.device).unsqueeze(0),
                contact_state=torch.as_tensor(contact_state, dtype=torch.float32, device=self.device).unsqueeze(0),
                morphology=torch.as_tensor(morphology, dtype=torch.float32, device=self.device).unsqueeze(0),
            )
        raw_vector = prediction.squeeze(0).detach().cpu().numpy().astype(float).tolist()
        decoded = decode_feagine_action(raw_vector, section_count=self.section_count)
        robot_state = observation.get("robot_state", {})
        contact_force = float(contact.get("max_force", 0.0)) if isinstance(contact, Mapping) else 0.0
        penetration = float(contact.get("max_penetration", 0.0)) if isinstance(contact, Mapping) else 0.0
        safe_action, safety = self.safety_projector.project(
            decoded,
            contact_force=contact_force,
            penetration=penetration,
            current_robot_state=robot_state if isinstance(robot_state, Mapping) else None,
            safety_mode="hold_current",
        )
        return safe_action, {
            "source": "adapter_policy",
            "raw_action_vector": raw_vector,
            "decoded_action": decoded,
            "safe_action": safe_action,
            "safety": safety,
        }

    @staticmethod
    def _fit(values: np.ndarray, size: int) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat.size == size:
            return flat
        if flat.size > size:
            return flat[:size]
        return np.pad(flat, (0, size - flat.size))
