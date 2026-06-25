from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

def feather(mask: np.ndarray, sigma: float = 6.0) -> np.ndarray:
  return np.clip(gaussian_filter(mask.astype(np.float64), sigma), 0.0, 1.0)

def composite(moving: np.ndarray, plate: np.ndarray, alpha: np.ndarray) -> np.ndarray:
  a = alpha[..., None]
  return a * moving + (1.0 - a) * plate