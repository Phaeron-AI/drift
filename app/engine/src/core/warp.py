from __future__ import annotations

import numpy as np

from .sampling import base_grid, sample

def backward_warp(image: np.ndarray, disp: np.ndarray, mode: str = "reflect") -> np.ndarray:
  height, width = image.shape[:2]
  grid = base_grid(height, width)
  src = grid + disp
  coords = np.stack([src[..., 0], src[..., 1]], axis=0)
  return sample(image, coords, mode=mode)