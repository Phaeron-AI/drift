from __future__ import annotations

import numpy as np
from typing import Optional

from .sampling import base_grid

def zero_field(height: int, width: int) -> np.ndarray:
  return np.zeros((height, width, 2), dtype=np.float64)

def constant_field(height: int, width: int, vy: float, vx: float) -> np.ndarray:
  M = zero_field(height, width)
  M[..., 0] = vy
  M[..., 1] = vx
  return M

def rotational_field(height: int, width: int, omega: float = 0.02, center: Optional[tuple[float, float]] = None) -> np.ndarray:
  cy, cx = center if center is not None else ((height - 1) / 2.0, (width - 1) / 2.0)
  G = base_grid(height, width)
  yy = G[..., 0] - cy
  xx = G[..., 1] - cx
  M = zero_field(height, width)
  M[..., 0] = omega * xx      # vy =  omega · x
  M[..., 1] = -omega * yy     # vx = -omega · y
  return M

def sway_field(height: int, width: int, amplitude: float = 4.0, wavelength: float = 120.0, phase: float = 0.0)-> np.ndarray:
  G = base_grid(height, width)
  y = G[..., 0]
  M = zero_field(height, width)
  M[..., 1] = amplitude * np.sin(2.0 * np.pi * y / wavelength + phase)
  return M

def apply_region(M: np.ndarray, region: np.ndarray) -> np.ndarray:
  return M * region[..., None]

def flow_field(height: int, width: int, angle_deg: float = 90.0,
               speed: float = 4.0) -> np.ndarray:
  theta = np.deg2rad(angle_deg)
  M = zero_field(height, width)
  M[..., 0] = speed * np.sin(theta)   # vy
  M[..., 1] = speed * np.cos(theta)   # vx
  return M