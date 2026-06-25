from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

Image = np.ndarray
Mask = np.ndarray
MotionField = np.ndarray
Displacement = np.ndarray


def validate_image(img: Image) -> None:
  dtype_str = str(getattr(img, 'dtype', ''))
  assert 'float' in dtype_str, f"Contract violation: Expected float dtype, got {dtype_str}"

  shape = getattr(img, 'shape', tuple())
  assert len(shape) == 3, f"Contract violation: Expected (H, W, 3), got shape {shape}"
  assert shape[-1] == 3, f"Contract violation: Expected 3 channels last, got shape {shape}"

  try:
    v_min = float(img.min())
    v_max = float(img.max())
  except (AttributeError, TypeError):
    raise AssertionError(f"Contract violation: Unable to compute min/max on object of type {type(img)}")

  assert v_min >= -1e-4, f"Contract violation: Image min value {v_min} is below expected range ~[0, 1]"
  assert v_max <= 1.0001, f"Contract violation: Image max value {v_max} is above expected range ~[0, 1]"


@dataclass
class Region:
  name: str
  mask: Mask
  strength: float = 1.0
  is_locked: bool = False
  # speed: float = 10.0
  # direction: str = ""


@dataclass
class EditSession:
  image: Image
  regions: list[Region] = field(default_factory=list)

  def animate_mask(self) -> Mask:
    active_masks = [r.mask for r in self.regions if not r.is_locked]

    if not active_masks:
      if self.regions:
        return self.regions[0].mask * 0.0
      raise RuntimeError("Cannot compute animate_mask: No regions exist to infer shape.")

    union_mask = active_masks[0]
    for mask in active_masks[1:]:
        union_mask = union_mask + mask

    if hasattr(union_mask, 'clamp'):
      return union_mask.clamp(0.0, 1.0)   # type: ignore
    elif hasattr(union_mask, 'clip'):
      return union_mask.clip(0.0, 1.0) 
    raise TypeError(f"Mask object of type {type(union_mask)} does not support clip or clamp.")


@dataclass
class Cinemagraph:
  frames: list[Image]
  fps: float
  seamless: bool


class Segmenter(Protocol):
  def segment(self, image: Image) -> list[Region]: ...


class MotionEstimator(Protocol):
  def estimate(self, image: Image, region: Region) -> MotionField: ...


class DisocclusionFiller(Protocol):
  def fill(self, warped: Image, plate: Image, hole_alpha: Mask) -> Image: ...