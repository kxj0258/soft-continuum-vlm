from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from soft_continuum_vlm.envs.mock_env import MockContinuumEnv
from soft_continuum_vlm.models.soft_embodiment_adapter import SoftEmbodimentAdapter
from soft_continuum_vlm.policies.adapter_policy import AdapterPolicy


def test_adapter_policy_loads_checkpoint_decodes_and_projects_action(tmp_path) -> None:
    checkpoint = tmp_path / "adapter.pt"
    model = SoftEmbodimentAdapter(
        vision_language_dim=64,
        proprioception_dim=8,
        contact_dim=3,
        morphology_dim=4,
        hidden_dim=8,
        action_dim=8,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"hidden_dim": 8},
            "input_dims": {
                "vision_language_dim": 64,
                "proprioception_dim": 8,
                "contact_dim": 3,
                "morphology_dim": 4,
                "action_dim": 8,
            },
            "action_schema": ["section_angles", "grip_command", "grasper_rotation"],
        },
        checkpoint,
    )
    observation = MockContinuumEnv(task="pick_red_object").reset(seed=0)

    action, info = AdapterPolicy(checkpoint).act(observation)

    assert set(action) == {"section_angles", "grip_command", "grasper_rotation"}
    assert info["source"] == "adapter_policy"
    assert len(info["raw_action_vector"]) == 8
    assert set(info) >= {"decoded_action", "safe_action", "safety"}
