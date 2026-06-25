from __future__ import annotations

from typing import Protocol

import numpy as np

class Backend(Protocol):
  def sample(self, field: np.ndarray, coords_yx: np.ndarray, mode: str)-> np.ndarray: ...
  def base_grid(self, height: int, width: int)-> np.ndarray: ...
