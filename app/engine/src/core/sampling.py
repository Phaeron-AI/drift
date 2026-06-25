from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

def base_grid(height: int, width: int)-> np.ndarray:
  ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
  return np.stack([ys, xs], axis=-1).astype(np.float64)

def sample(field: np.ndarray, coords_yx: np.ndarray, mode: str = "reflect") -> np.ndarray:
  if field.ndim == 2:
    return map_coordinates(field, coords_yx, order=1, mode=mode)
  channels = [
    map_coordinates(field[..., c], coords_yx, order=1, mode=mode)
    for c in range(field.shape[-1])
  ]
  return np.stack(channels, axis=-1)

