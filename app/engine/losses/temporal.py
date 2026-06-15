from __future__ import annotations

import torch
from torch import Tensor

def temporal_consistency_loss(pred_frames: Tensor, target_frames: Tensor)-> Tensor:
  if pred_frames.shape[1] < 2:
    return pred_frames.new_zeros(())
  
  pred_delta = pred_frames[:, 1:] - pred_frames[:, :-1]      # [B, F-1, C, h, w]
  true_delta = target_frames[:, 1:] - target_frames[:, :-1]
  return ((pred_delta - true_delta) ** 2).mean()