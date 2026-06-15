from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

def mask_aware_reconstruction_loss(
  noise_pred: Tensor,        # [B, C, h, w]
  noise_target: Tensor,      # [B, C, h, w]
  mask: Tensor,              # [B, 1, H, W]  
  mask_weight: float = 4.0,  # lambda: how much more the object/hole region counts
) -> Tensor:

  hw = noise_pred.shape[-2:]
  if mask.shape[-2:] != hw:
    mask = F.interpolate(mask.to(noise_pred.dtype), size=hw, mode="area")
  else:
    mask = mask.to(noise_pred.dtype)

  # Build the Weight Map: w(x) = 1 + lambda * m(x)
  weight_map = 1.0 + (mask_weight * mask) # [B, 1, h, w]

  squared_error = (noise_pred - noise_target) ** 2  # Shape: [B, C, h, w]

  channels = squared_error.shape[1]
  loss = (weight_map * squared_error).sum() / (weight_map.sum() * channels)

  return loss