from __future__ import annotations

import numpy as np

from ..core import sampling

class NumpyBackend:
  name = "numpy"

  def sample(self, field: np.ndarray, coords_yx: np.ndarray, mode: str = "reflect")-> np.ndarray:
    return sampling.sample(field, coords_yx, mode)
  
  def base_grid(self, height: int, width: int)-> np.ndarray:
    return sampling.base_grid(height, width)