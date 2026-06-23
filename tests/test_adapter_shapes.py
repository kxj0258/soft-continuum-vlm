import pytest

torch = pytest.importorskip("torch")

from soft_continuum_vlm.models.soft_embodiment_adapter import SoftEmbodimentAdapter


def test_soft_embodiment_adapter_outputs_continuum_action_shape() -> None:
    model = SoftEmbodimentAdapter(
        vision_language_dim=16,
        proprioception_dim=8,
        contact_dim=4,
        morphology_dim=6,
        hidden_dim=32,
        action_dim=8,
    )

    action = model(
        vision_language_feature=torch.zeros(3, 16),
        proprioception=torch.zeros(3, 8),
        contact_state=torch.zeros(3, 4),
        morphology=torch.zeros(3, 6),
    )

    assert action.shape == (3, 8)
