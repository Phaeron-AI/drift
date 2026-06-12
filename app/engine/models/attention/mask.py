from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MaskToBias(nn.Module):
  def __init__(self, latent_size: int, n_heads: int, n_text: int, n_traj: int)-> None:
    super().__init__()
    self.latent_size = latent_size
    self.n_heads = n_heads
    self.n_text = n_text
    self.n_traj = n_traj

    self.proj = nn.Linear(1, self.n_heads)

    nn.init.zeros_(self.proj.weight)
    nn.init.zeros_(self.proj.bias)
  
  def forward(self, mask: Tensor)-> Tensor:
    B, _, _, _ = mask.shape
    S_q = self.latent_size * self.latent_size
    S_kv = self.n_text + self.n_traj

    latent_mask = F.interpolate(
      mask,
      size=(self.latent_size, self.latent_size),
      mode="area"
    )

    latent_mask_flat = latent_mask.view(B, 1, S_q).transpose(1, 2)

    head_bias = self.proj(latent_mask_flat).permute(0, 2, 1).unsqueeze(-1)

    full_bias = torch.zeros(
      (B, self.n_heads, S_q, S_kv), 
      device=mask.device, 
      dtype=mask.dtype
    )

    full_bias[:, :, :, self.n_text : self.n_text + self.n_traj] = head_bias

    return full_bias