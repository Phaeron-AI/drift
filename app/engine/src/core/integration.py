from __future__ import annotations

import numpy as np

from .sampling import base_grid, sample

def integrate_series(M: np.ndarray, n_frames: int, sign: float = 1.0) -> list[np.ndarray]:
  height, width = M.shape[:2]
  grid = base_grid(height, width)
  velocity = sign * M

  disp = np.zeros_like(M)
  series = [disp.copy()]
  for _ in range(n_frames):
    pos = grid + disp                                  # p + F_{0→t}(p)
    coords = np.stack([pos[..., 0], pos[..., 1]], axis=0)
    v_here = sample(velocity, coords)                  # M at displaced position
    disp = disp + v_here
    series.append(disp.copy())
  return series