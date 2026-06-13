from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .trajectory_encoder import TrajectoryEncoder
from .mask import MaskToBias

class SpatialCrossAttention(nn.Module):
  def __init__(self, inner_dim: int, context_dim: int = 768, n_heads: int = 8, latent_size: int = 64, n_traj: int = 1, n_text: int = 77)-> None:
    super().__init__()
    self.inner_dim = inner_dim
    self.head_dim = inner_dim // n_heads
    self.n_heads = n_heads
    self.latent_size = latent_size
    self.n_traj = n_traj
    self.n_text = n_text

    self.to_q = nn.Linear(inner_dim, inner_dim, bias=False)
    self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
    self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
    self.to_o = nn.Linear(inner_dim, inner_dim)

    self.traj_encoder = TrajectoryEncoder(embed_dim=context_dim, n_tokens=self.n_traj)
    self.mask_to_bias = MaskToBias(latent_size=self.latent_size, n_heads=self.n_heads, n_traj=self.n_traj)
  
  def _to_heads(self, x: Tensor)-> Tensor:
    B, S, _ = x.shape # [B, S, inner_dim]
    x = x.view(B, S, self.n_heads, self.head_dim).transpose(1, 2) # [B, S, n_heads, head_dim]^T
    return x
  
  def _merge_heads(self, x: Tensor)-> Tensor:
    B, _, S_q, _ = x.shape  # [B, n_heads, S_q, head_dim]
    x = x.transpose(1, 2)

    return x.reshape(B, S_q, self.inner_dim) # [B, s_q, inner_dim]
  
  def forward(self, hidden_states: Tensor, trajectory: Tensor, mask: Tensor, text_embeds: Optional[Tensor] = None)-> Tensor:
    traj_token = self.traj_encoder(trajectory)  # [B, n_traj, E]
    if text_embeds is not None:
      context = torch.cat([text_embeds, traj_token], dim=1) # [B, n_text x n_traj, E]
      n_text_eff = self.n_text
    else:
      context = traj_token
      n_text_eff = 0
    
    q = self._to_heads(self.to_q(hidden_states))  # [B, S_q, inner_dim]
    k = self._to_heads(self.to_k(context))  # [B, S_k, inner_dim]
    v = self._to_heads(self.to_v(context))  # [B, S_v, inner_dim]

    bias = self.mask_to_bias(mask, n_text_eff).to(q.dtype)  # [B, n_heads, S_q, S_kv]

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias) #   SHAPE: q[B,h,S_q,hd], k/v[B,h,S_kv,hd], bias[B,h,S_q,S_kv] -> out [B,h,S_q,hd]
    out = self._merge_heads(out)

    return self.to_o(out)