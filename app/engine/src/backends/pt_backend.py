from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from typing import Union

from ..core import sampling

class TorchBackend:
  name = "torch"

  def __init__(self, height: int, width: int, device: str = "cuda")-> None:
    self.device = device

    self._grids: dict[tuple[int, int], torch.Tensor] = {}
  
  def sample(self, field: np.ndarray, coords_yx: np.ndarray, mode: str = "reflect")-> Union[np.ndarray, Tensor]:
    is_numpy = isinstance(field, np.ndarray)

    if is_numpy:
      field_t = torch.from_numpy(field).float().to(self.device)
      coords_t = torch.from_numpy(coords_yx).float().to(self.device)
    else:
      field_t = field.to(self.device)
      coords_t = coords_yx.to(self.device)  # type: ignore
    
    if field_t.ndim == 3:
      field_t = field_t.unsqueeze(0)
    
    _, _, H, W = field_t.shape # [N, C, H, W]

    y_coords = coords_t[0]
    x_coords = coords_t[1]

    gx = 2.0 * x_coords / max(W - 1, 1) - 1.0
    gy = 2.0 * y_coords / max(H - 1, 1) - 1.0

    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0) # [1, H, W, 2]

    pad_mode = 'reflection' if mode == 'reflect' else mode

    out = F.grid_sample(
      field_t,
      grid,
      mode="bilinear",
      padding_mode=pad_mode,
      align_corners=True
    )

    out = out.squeeze(0)

    if is_numpy:
      return out.cpu().numpy()
    
    return out
  
  def base_grid(self, height: int, width: int)-> Union[np.ndarray, Tensor]:
    key = (height, width)
    if key not in self._grids:
      y = torch.arange(height, device=self.device, dtype=torch.float32)
      x = torch.arange(width, device=self.device, dtype=torch.float32)
      
      # indexing='ij' ensures output is strictly (y, x) order
      gy, gx = torch.meshgrid(y, x, indexing='ij')
      
      self._grids[key] = torch.stack([gy, gx], dim=0) # [2, H, W]
        
    return self._grids[key]
