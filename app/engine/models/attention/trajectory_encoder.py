from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class TrajectoryEncoder(nn.Module):
  def __init__(self, embed_dim: int = 768, num_freqs: int = 8, n_tokens: int = 1) -> None:
    super().__init__()
    self.embed_dim = embed_dim
    self.num_freqs = num_freqs
    self.n_tokens = n_tokens

    in_features = 4 * num_freqs
    out_features = embed_dim * n_tokens

    self.mlp = nn.Sequential(
      nn.Linear(in_features, embed_dim),
      nn.GELU(),
      nn.Linear(embed_dim, out_features),
    )

  def forward(self, trajectory: Tensor) -> Tensor:
    B, _ = trajectory.shape  # [B, 2]
    device = trajectory.device

    traj = trajectory.float()
    freq_bands = (2.0 ** torch.arange(self.num_freqs, device=device, dtype=torch.float32)) * math.pi
    scaled = traj.unsqueeze(-1) * freq_bands.view(1, 1, -1)

    sin_feat = torch.sin(scaled)
    cos_feat = torch.cos(scaled)
    features = torch.cat([sin_feat, cos_feat], dim=-1).view(B, -1)  # float32

    encoded_flat = self.mlp(features.to(self.mlp[0].weight.dtype))  # type: ignore
    tokens = encoded_flat.view(B, self.n_tokens, self.embed_dim)

    return tokens