from __future__ import annotations

import torch
from torch import nn


class SoftEmbodimentAdapter(nn.Module):
    """Small MLP stub that maps multimodal state features to continuum actions."""

    def __init__(
        self,
        *,
        vision_language_dim: int,
        proprioception_dim: int,
        contact_dim: int,
        morphology_dim: int,
        hidden_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        input_dim = vision_language_dim + proprioception_dim + contact_dim + morphology_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        *,
        vision_language_feature: torch.Tensor,
        proprioception: torch.Tensor,
        contact_state: torch.Tensor,
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [vision_language_feature, proprioception, contact_state, morphology],
            dim=-1,
        )
        return self.network(features)
